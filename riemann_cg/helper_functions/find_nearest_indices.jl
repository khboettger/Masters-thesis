## Code to find the nearest indices in some Array to some value

function find_nearest_indices(
    arr::Vector{Float64}, 
    X::Float64, 
    N::Int
    )
    
    n = length(arr)

    # Edge cases: X is out of bounds
    if X <= arr[1]
        return collect(1:min(N, n))
    elseif X >= arr[end]
        return collect(max(1, n - N + 1):n)
    end

    # Find insertion point (not necessarily exact match)
    idx = searchsortedfirst(arr, X)

    # Initialize two pointers around insertion index
    left = idx - 1
    right = idx

    result = Int[]

    while length(result) < N
        if left < 1
            push!(result, right)
            right += 1
        elseif right > n
            push!(result, left)
            left -= 1
        else
            if abs(arr[left] - X) <= abs(arr[right] - X)
                push!(result, left)
                left -= 1
            else
                push!(result, right)
                right += 1
            end
        end
    end

    sort(result)  # Optional: sort the indices
end
