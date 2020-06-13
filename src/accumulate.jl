# scan and accumulate

## COV_EXCL_START

# partial scan of individual thread blocks within a grid
# Parallel Prefix Scan using warp-level primitives
#
# https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
# 
#
# performance TODOs:
# - Multiple elements per thread 
# - Use warpSize instead of hardcoding and move to GPUArrays
function partial_scan(op::Function, output::CuDeviceArray{T}, input::CuDeviceArray, aggregates::CuDeviceArray,
                        Rdim, Rpre, Rpost, Rother, neutral) where {T}
    threads = blockDim().x
    thread = threadIdx().x
    warps = threads ÷ 32
    mask = typemax(UInt32)

    temp = @cuDynamicSharedMem(T, (threads,))

    # iterate the main dimension using threads and the first block dimension
    tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # iterate the other dimensions using the remaining block dimensions
    bid = (blockIdx().z-1) * gridDim().y + blockIdx().y

    lane_id = (threadIdx().x - 1) % 32
    warp_id = (threadIdx().x - 1) ÷ 32

    if bid > length(Rother)
        return
    end

    @inbounds begin
        I = Rother[bid]
        Ipre = Rpre[I[1]]
        Ipost = Rpost[I[2]]
    end

    # load input into shared memory (apply `op` to have the correct type)
    value = op(neutral, neutral)
    if tid <= length(Rdim)
        @inbounds value = input[Ipre, tid, Ipost]
    else
        return
    end

    sync_threads()

    # Compute sums within thread warp
    i = 1
    while i <= 32
        n = shfl_up_sync(mask, value, i)
        if lane_id >= i
            value = op(value, n)
        end
        i *= 2
    end

    # Write warp sum to shared memory
    if lane_id == 31
        @inbounds temp[warp_id + 1] = value
    end
    sync_threads()

    # Warp 0 computes intermediate sums
    if warp_id == 0
        warp_sum = op(neutral, neutral)

        if lane_id < warps
            warp_sum = temp[lane_id + 1]
        end

        i = 1
        while i <= 32
            n = shfl_up_sync(mask, warp_sum, i)
            if lane_id >= i
                warp_sum = op(warp_sum, n)
            end
            i *= 2
        end
        warp_sum = shfl_up_sync(mask, warp_sum, 1)
        if lane_id == 0
            warp_sum = op(neutral, neutral)
        end

        if lane_id < warps
            @inbounds temp[lane_id + 1] = warp_sum
        end

    end
    sync_threads()



    @inbounds begin
        value = op(temp[warp_id + 1], value)
        output[Ipre, tid, Ipost] = value
        if threadIdx().x == blockDim().x || tid == length(Rdim)
            aggregates[Ipre, blockIdx().x, Ipost] = value
        end
    end

    return
end

# aggregate the result of a partial scan by applying preceding block aggregates
function aggregate_partial_scan(op::Function, output::CuDeviceArray,
                                aggregates::CuDeviceArray,
                                Rdim, Rpre, Rpost, Rother, init, neutral, ::Val{inclusive}=Val{true}) where inclusive

    # iterate the main dimension using threads and the first block dimension
    tid = (blockIdx().x-1) * blockDim().x + threadIdx().x
    # iterate the other dimensions using the remaining block dimensions
    bid = (blockIdx().z-1) * gridDim().y + blockIdx().y

    block = (tid - 1) ÷ 256

    if tid > length(Rdim)
        return
    end

    value = neutral
    if  init !== nothing
        value = op(init, value)
    end

    @inbounds begin
        I = Rother[bid]
        Ipre = Rpre[I[1]]
        Ipost = Rpost[I[2]]
        if block != 0
            value = op(value, aggregates[Ipre, block, Ipost])
        end
        if inclusive
            output[Ipre, tid, Ipost] = op(value, output[Ipre, tid, Ipost])
        else
            if threadIdx().x != 1
                value = op(value, output[Ipre, tid - 1, Ipost])
            end
            output[Ipre, tid, Ipost] = value
        end
    end

    return
end

## COV_EXCL_STOP

function scan!(f::Function, output::CuArray{T}, input::CuArray;
    inclusive = true, dims::Integer = 1, init=nothing, 
    neutral=GPUArrays.neutral_element(f, T)) where {T}
    dims > 0 || throw(ArgumentError("dims must be a positive integer"))
    inds_t = axes(input)
    axes(output) == inds_t || throw(DimensionMismatch("shape of B must match A"))
    dims > ndims(input) && return copyto!(output, input)
    isempty(inds_t[dims]) && return output

    f = cufunc(f)

    # iteration domain across the main dimension
    Rdim = CartesianIndices((size(input, dims),))

    # iteration domain for the other dimensions
    Rpre = CartesianIndices(size(input)[1:dims-1])
    Rpost = CartesianIndices(size(input)[dims+1:end])
    Rother = CartesianIndices((length(Rpre), length(Rpost)))

    elementsPerThread = 1
    threadsPerBlock = 256;

    #round threads to neareRst multiple of 32temp[warp_id + 1]
    threads = ((length(Rdim)-1)÷32+1)*32
    blocks_x = (threads - 1) ÷ threadsPerBlock + 1 

    #Determine Grid Dims
    blocks_others = (length(Rpre), length(Rpost))

    # Declare aggregates for recursively computing large arrays
    aggregates_dims = (size(input)[1:(dims - 1)]...,blocks_x, size(input)[(dims+1):end]...)
    aggregates = CuArray{T}(undef, aggregates_dims)

    # Compute partial scan
    args = (f, output, input, aggregates, Rdim, Rpre, Rpost, Rother, neutral)
    block_dims = (blocks_x, blocks_others...)

    @cuda(threads=threadsPerBlock, blocks=block_dims, shmem =sizeof(T)*threadsPerBlock, partial_scan(args...))

    # Recusively compute partial sum of partial sums
    if length(Rdim) > threadsPerBlock
        accumulate!(f, aggregates, aggregates, dims=dims)
    end

    # Broadcast partial sums
    args = (f, output, aggregates, Rdim, Rpre, Rpost, Rother, init, neutral, Val(inclusive))
    @cuda(threads=threadsPerBlock, blocks=block_dims, aggregate_partial_scan(args...))

    unsafe_free!(aggregates)
    return output
end


## Base interface

Base._accumulate!(op, output::CuArray, input::CuVector, dims::Nothing, init::Nothing) =
    scan!(op, output, input; dims=1)

Base._accumulate!(op, output::CuArray, input::CuArray, dims::Integer, init::Nothing) =
    scan!(op, output, input; dims=dims)

Base._accumulate!(op, output::CuArray, input::CuVector, dims::Nothing, init::Some) =
    scan!(op, output, input; dims=1, init=init)

Base._accumulate!(op, output::CuArray, input::CuArray, dims::Integer, init::Some) =
    scan!(op, output, input; dims=dims, init=init)

Base.accumulate_pairwise!(op, result::CuVector, v::CuVector) = accumulate!(op, result, v)
