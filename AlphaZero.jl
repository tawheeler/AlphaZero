
using StatsBase

const State = UInt32 # bits 1-9 are x, bits 10-8 are o. You can tell whose turn it is by number set
                     # bit 1 is top left, bit 2 is center left, bit 3 is bottom left, bit 4 is top center:
                     #    1  4  7
                     #    2  5  8
                     #    3  6  9

is_x_at(s::State, loc::Int) = (s & (one(UInt32) << (loc-1))) > 0
is_o_at(s::State, loc::Int) = (s & (one(UInt32) << (loc+8))) > 0
is_any_at(s::State, loc::Int) = is_x_at(s, loc) || is_o_at(s, loc)
char_at(s::State, loc::Int) = is_x_at(s, loc) ? "x" : is_o_at(s, loc) ? "o" : " "

place_x_at(s::State, loc::Int) = s ⊻ (one(UInt32) << (loc-1))
place_o_at(s::State, loc::Int) = s ⊻ (one(UInt32) << (loc+8))

n_fields_set(s::State) = count_ones(s)
black_to_move(s::State) = mod(n_fields_set(s),2) == 0

function next_state(s::State, loc::Int)
    if black_to_move(s)
        return place_x_at(s, loc)
    else
        return place_o_at(s, loc)
    end
end

function black_wins(s::State)
    for mask in (0x00000007, 0x00000038, 0x000001c0, 0x00000049, 0x00000092, 0x00000124, 0x00000111, 0x00000054)
        if s & mask == mask
            return true
        end
    end
    return false
end
function white_wins(s::State)
    for mask in (0x00000007, 0x00000038, 0x000001c0, 0x00000049, 0x00000092, 0x00000124, 0x00000111, 0x00000054)
        white_mask = mask << 9
        if s & white_mask == white_mask
            return true
        end
    end
    return false
end
game_over(s::State) = black_wins(s) || white_wins(s) || n_fields_set(s) == 9
function game_status(s::State)
    if black_wins(s)
        return 1
    elseif white_wins(s)
        return -1
    else
        return 0
    end
end


function valid_moves(s::State)
    retval = Int[]
    if !game_over(s)
        for i in 1 : 9
            if !is_any_at(s, i)
                push!(retval, i)
            end
        end
    end
    return retval
end

function print_board(io::IO, s::State)
    println(io, "game_status:   ", game_status(s))
    println(io, "black_to_move: ", black_to_move(s))
    @printf(io, " %s | %s | %s\n", char_at(s,1), char_at(s,4), char_at(s,7))
    println(io, "--- --- ---")
    @printf(io, " %s | %s | %s\n", char_at(s,2), char_at(s,5), char_at(s,8))
    println(io, "--- --- ---")
    @printf(io, " %s | %s | %s\n", char_at(s,3), char_at(s,6), char_at(s,9))
end
print_board(s::State) = print_board(STDOUT, s)

mutable struct MCTSNode

    s::State

    N::Int # visit counts
    W::Float64 # total action value
    Q::Float64 # mean action value
    P::Float64 # predicted probabiliy of this being the successive action

    parent::Union{Void,MCTSNode}
    children::Vector{MCTSNode} # always 9 children (I know, inefficient....)
end

MCTSNode() = MCTSNode(zero(UInt32), 0, 0.0, 0.0, 1.0, nothing, MCTSNode[])
isleaf(node::MCTSNode) = isempty(node.children)
isroot(node::MCTSNode) = isa(node.parent, Void)

function mcts_select_action(node::MCTSNode, c::Float64)

    v_best = -Inf
    a_best = 0

    sqrt_term = sqrt(sum(b.N for b in node.children))
    sgn = black_to_move(node.s) ? 1 : -1

    for a in 1 : 9
        child = node.children[a]
        U = c*child.P*sqrt_term/(1 + child.N)
        v = sgn*child.Q + U

        if v > v_best && child.P > 0
            v_best, a_best = v, a
        end
    end

    return a_best
end

function mcts_select(root::MCTSNode,
    c::Float64, # constant which determines exploration / exploitation tradeoff
    )

    #=
    The first in-tree phase of each sim begins at the root node
<<<<<<< HEAD
    and finishes when the sim reaches a lead node. 
=======
    and finishes when the sim reaches a lead node.
>>>>>>> 426423c567dfd1e893d1704e43a0f9dad95ca665
    Actions are selected according to the PUCT algorithm.
    The strategy initially prefers actions with with prior prob and low visit count,
    but asymptotically perfers actions with high action value.
    =#

    node = root
    while !isleaf(node)
        a = mcts_select_action(node, c)
        node = node.children[a]
    end
    return node
end
function mcts_expand!(leaf::MCTSNode, p::Vector{Float64})
    #=
    The leaf is evaluated by the NN, but is first transformed by a random
    dihedral reflection or rotation selected unformly at random from 1:8
    The leaf node is expanded and each edge is initialized with:
    N = 0, W = 0, Q = 0, P = pₐ
    The value v is then backed up.

    I assume evaluation already happened.
    p is the probability distribution from the neural net
    with invalid actions already zeroed out,
    probabilities already normalized back to sum to one

    This will expand the tree irrespective of whether any legal moves exist
    =#

    append!(leaf.children, [MCTSNode(next_state(leaf.s, i), 0, 0.0, 0.0, p[i], leaf, MCTSNode[]) for i in 1:length(p)])
    return leaf
end
function mcts_backup!(leaf::MCTSNode, v::Float64)
    #=
    The edge stats are updated in a backward pass up the tree.
    The visit counts are incremented.
    The action value is updated to the mean value.

    Here, v was predicted by the net at the leaf.
    =#

    node = leaf
    while !isroot(node)
        node.N += 1
        node.W += v
        node.Q = node.W / node.N
        node = node.parent
    end

    # update root as well
    node.N += 1
    node.W += v
    node.Q = node.W / node.N

    return node
end

function mcts_sim!(root::MCTSNode, c::Float64)

    leaf = mcts_select(root, c)

    if game_over(leaf.s)
        v = convert(Float64, game_status(leaf.s))
        mcts_backup!(leaf, v)
    else
        # p, v = predict(s)
        
        p = ones(Float64, 9)
        v = 0.0 # no clue who wins

        # zero-out invalid moves
        for i in 1 : 9
            if is_any_at(leaf.s, i)
                p[i] = 0.0
            end
        end

        @assert sum(p) > 0

        # produce valid prob distr
        normalize!(p, 1)

        mcts_expand!(leaf, p)
        mcts_backup!(leaf, v)
    end

    return root
end
function run_mcts!(root::MCTSNode, c::Float64, nsims::Int)
    for i in 1 : nsims
        mcts_sim!(root, c)
    end
    return root
end

function get_policy_probabilities!(π::Vector{Float64}, root::MCTSNode, τ::Float64)
    for (i,child) in enumerate(root.children)
        π[i] = child.N^(1/τ)
    end
    normalize!(π, 1)
    return π
end
draw_action(π::Vector{Float64}) = sample(1:9, ProbabilityWeights(π))

function display_tree(node::MCTSNode, ntabs::Int=0)
    @printf("%s%4d  %8.3f  %8.3f  %8.3f\n", "\t"^ntabs, node.N, node.W, node.Q, node.P)
    for child in node.children
        display_tree(child, ntabs+1)
    end
end

# struct Model
#     x::Vector{Float64}

#     W::Matrix{Float64}
#     b::Vector{Float64}
#     # relu for 1:9, tanh for 10 (value)

#     y::Vector{Float64}

#     z::Float64
#     π::Vector{Float64}
#     c::Float64
#     loss::Float64
# end
# relu(x::Real) = max(0,x)
# function forward!(M::Model, s::State)

#     x, W, b, y = M.x, M.W, M.b, M.y
#     for i in 1 : 8
#         x[i] = is_x_at(s, i)
#         x[i+8] = is_o_at(s, i)
#     end

#     y[:] = W*x + b
#     y[1:9] = relu.(y[1:9])
#     y[10] = tanh(y[10])

#     return M
# end
# function propagate_through_loss!(M::Model, z::Float64, π::Vector{Float64}, c::Float64)
#     M.z = z
#     copy!(M.π, π)
#     M.c = c
#     M.loss = (z - M.y[10])
# end
# function backward!(M::Model, ∇W::Matrix{Float64}, ∇b::Vector{Float64})
# end


using Base.Test

let
    s = zero(UInt32)
    @test black_to_move(s)
    @test !black_wins(s)
    @test !white_wins(s)
    @test !is_x_at(s, 1)
    @test !is_o_at(s, 1)
    @test valid_moves(s) == [1,2,3,4,5,6,7,8,9]

    s = place_x_at(s, 1)
    @test !black_to_move(s)
    @test !black_wins(s)
    @test !white_wins(s)
    @test  is_x_at(s, 1)
    @test !is_o_at(s, 1)
    @test valid_moves(s) == [2,3,4,5,6,7,8,9]

    s = place_o_at(s, 4)
    s = place_x_at(s, 2)
    s = place_o_at(s, 5)
    @test !black_wins(s)
    @test !white_wins(s)
    @test game_status(s) == 0
    @test valid_moves(s) == [3,6,7,8,9]

    s = place_x_at(s, 3)
    @test  black_wins(s)
    @test !white_wins(s)
    @test game_status(s) == 1
    @test valid_moves(s) == Int[]

    s = place_o_at(s, 6)
    @test  white_wins(s)
    @test valid_moves(s) == Int[]
end

let
    root = MCTSNode()
    @test mcts_select(root, 1.0) === root

    mcts_sim!(root, 1.0)
    mcts_sim!(root, 1.0)

    π = zeros(Float64, 9)
    @test get_policy_probabilities!(π, root, 1.0) == [1,0,0,0,0,0,0,0,0]

    mcts_sim!(root, 1.0)
    mcts_sim!(root, 1.0)

    for i in 1 : 1000
        mcts_sim!(root, 1.0)
        get_policy_probabilities!(π, root, 1.0)
    end
end

let
    s = zero(UInt32)
    s = place_x_at(s, 1)
    s = place_x_at(s, 2)
    s = place_o_at(s, 4)
    s = place_o_at(s, 5)
    root = MCTSNode(s, 0, 0.0, 0.0, 1.0, nothing, MCTSNode[])

    π = zeros(Float64, 9)
    for i in 1 : 10
        mcts_sim!(root, 0.01)
    end

    get_policy_probabilities!(π, root, 1.0)
    @show π

    display_tree(root)
end

srand(0)
let
    root = MCTSNode()

    while !game_over(root.s)
        print_board(root.s)

        run_mcts!(root, 1.0, 100)
        π = zeros(Float64, 9)
        get_policy_probabilities!(π, root, 1.0)
        a = draw_action(π)
        root = root.children[a]
    end

    print_board(root.s)
end
