using AutomotiveDrivingModels
using AutoViz
using Reel
using Vec
using Distributions
using AdaptiveStressTesting
using Records
using Interact
using PyPlot

type SimParams
    endtime::Int64
    logging::Bool
end

type CarSim
    p::SimParams
    scene::Scene             # The scene contains onfo about cars
    roadway::CrosswalkEnv         # roadway, layout of the road
    models::Dict{Int, DriverModel}     # driver models for each car
    actions::Array{Any,1}     # actions for each car, MCTS controls actions[1]
    initial::Scene
    states::Vector{Scene}        # Array that holds all states
    t::Int64
end

function AutoViz.render!(rendermodel::RenderModel, env::CrosswalkEnv)
    render!(rendermodel, env.roadway)

    curve = env.crosswalk.curve
    n = length(curve)
    pts = Array(Float64, 2, n)
    for (i,pt) in enumerate(curve)
        pts[1,i] = pt.pos.x
        pts[2,i] = pt.pos.y
    end

    add_instruction!(rendermodel, render_dashed_line, (pts, colorant"white", env.crosswalk.width, 1.0, 1.0, 0.0, Cairo.CAIRO_LINE_CAP_BUTT))
    return rendermodel
end

function CarSim(params::SimParams, models::Dict{Int, DriverModel})

    roadway = gen_straight_roadway(2,100.0,origin = VecSE2(-50.0,0.0,0.0))

    cam = FitToContentCamera(0.0)
    crosswalk = Lane(LaneTag(2,1), gen_straight_curve(VecE2(0.0, -DEFAULT_LANE_WIDTH), VecE2(0.0, 2*DEFAULT_LANE_WIDTH), 2), width=2.0)
    cw_segment = RoadSegment(2, [crosswalk])
    push!(roadway.segments, cw_segment)
    sensor = SimpleSensor(0.1,0.1,0.0)
    env = CrosswalkEnv(roadway, crosswalk, sensor,Array{Any,1}())
    cw_roadway = Roadway([RoadSegment(2, [env.crosswalk])]);
    PEDESTRIAN_DEF = VehicleDef(AgentClass.PEDESTRIAN, 1.0, 1.0)
    scene = Scene()

    cars = [Vehicle(VehicleState(VecSE2(-35.0,0,0), roadway.segments[1].lanes[1],roadway, 11.17),
             VehicleDef(), 1), Vehicle(VehicleState(VecSE2(0.0,-4.0,π/2), env.crosswalk, cw_roadway, 1.4),
             PEDESTRIAN_DEF, 2)]
#         ,Vehicle(VehicleState(VecSE2(0.0,-1.0,π/2), env.crosswalk, cw_roadway, 1.4),
#              PEDESTRIAN_DEF, 3)]
    car1 = cars[1]
    car2 = cars[2]
#     car3 = cars[3]
    push!(scene, car1)
    push!(scene, car2)
    initial = deepcopy(scene)
    env.observed = [car2]
    actions = get_actions!(Array(Any, length(scene)), scene, roadway, models)
    CarSim(params, scene, env, models, actions, initial, Vector{Scene}(), 0)
end
# function initialize_ped(sim::CarSim)
#     cw_roadway = Roadway([RoadSegment(2, [sim.roadway.crosswalk])]);
#     PEDESTRIAN_DEF = VehicleDef(AgentClass.PEDESTRIAN, 1.0, 1.0)
#     id = sim.scene.n + 1
#     if rand() >= 0.5
#         ped = Vehicle(VehicleState(VecSE2(0.0,-2.0,π/2), sim.roadway.crosswalk, cw_roadway, 1.4),
#              PEDESTRIAN_DEF, id)
#     else
#         ped = Vehicle(VehicleState(VecSE2(0.0,7.0,-π/2), sim.roadway.crosswalk, cw_roadway, 1.4),
#              PEDESTRIAN_DEF, id)
#     end
#     return ped
# end
function initialize(sim::CarSim)
    sim.t = 0
    sim.scene = deepcopy(sim.initial)
    sim.roadway.observed = [sim.scene[2]]
    empty!(sim.states)
    if sim.p.logging
        push!(sim.states, deepcopy(sim.scene))
    end
end

function update1(sim::CarSim)
    sim.t += 1
    #  ped flow
#     k = sim.scene.n + 1
#     ped_entry_prob = 3^k * exp(-3)/factorial(k) # Parametrize λ
#     if rand() <= ped_entry_prob
#         new_ped = initialize_ped(sim)
#         push!(sim.scene,new_ped)
#         sim.models[new_ped.id] = CrosswalkDriver(LaneSpecificAccelLatLon(0.0,0.0), [.01, 0.1])
#         append!(sim.actions, 0.0)
#         append!(sim.roadway.observed, 0.0)
#         sim.roadway.observed[k-1] = new_ped
#     end
    get_actions!(sim.actions, sim.scene, sim.roadway, sim.models)
    tick!(sim.scene, sim.roadway.roadway, sim.actions, 0.1)
    prob = 0
    for i in 2:sim.scene.n
        prob = prob + sqrt(sqmahal(sim.models[i], sim.actions[i]))
    end
    prob = prob + sim.roadway.sensormodel.likelihood
    locs = [car.state.posG for car in sim.scene]
    dists = [dist(convert(VecE2, locs[1]), convert(VecE2, locs[i])) for i in 2:sim.scene.n]
    if sim.p.logging
        push!(sim.states, deepcopy(sim.scene))
    end
    return (prob, isevent(sim), minimum(dists))
end

# function update2(sim::CarSim)
#     sim.t += 1
#     lat = rand(Normal(0.0, 5.0))
#     lon = rand(Normal(0.0, 5.0))
#     get_actions!(sim.actions, sim.scene, sim.roadway.roadway, sim.models)
#     sim.actions[2] = LaneSpecificAccelLatLon(lat, lon)
#     tick!(sim.scene, sim.roadway.roadway, sim.actions, 0.1)
#     prob = pdf(Normal(0.0, 5.0), lat) * pdf(Normal(0.0, 5.0), lon)
#     locs = [car.state.posG for car in sim.scene]
#     dists = [dist(convert(VecE2, locs[1]), convert(VecE2, locs[i])) for i in 2:sim.scene.n]
#     if sim.p.logging
#         push!(sim.states, deepcopy(sim.scene))
#     end
#     return (prob, isevent(sim), minimum(dists))
# end

function isevent(sim::CarSim)
    event = false
    E = [is_colliding(sim.scene[1], sim.scene[i]) for i in 2:sim.scene.n]
    if any(x -> x > 0, E)
        event = true
    end
    return event
end

function isterminal(sim::CarSim)

    isevent(sim) || sim.t >= sim.p.endtime
end

function reward_fun(prob::Float64, event::Bool, terminal::Bool, dist::Float64,
                            ast::AdaptiveStressTest,sim::CarSim) #ast and sim not used in default
    r = -log(1 + prob)
    if event
        r += 0.0
    elseif terminal #incur distance cost only if !event && terminal
        r += -10000 - 1000*dist
    end
    r
end

function run_sim(maxtime::Int64,s::Array{Float64,1},d::Int64,n::Int64,seed::Int64)
    const MAXTIME = maxtime #sim endtime
    const RNG_LENGTH = 2
    timestep = 0.1
    models = Dict{Int, DriverModel}()
    models[1] = Tim2DDriver(timestep, mlon = IntelligentDriverModel(), mlat = ProportionalLaneTracker())
    models[2] = CrosswalkDriver(LaneSpecificAccelLatLon(0.0,0.0), s)

    sim_params = SimParams(MAXTIME, true)

    sim = CarSim(sim_params, models)

    ast_params = ASTParams(MAXTIME, RNG_LENGTH, seed, 0)
    ast = AdaptiveStressTest(ast_params, sim, initialize, update1, isterminal, reward_fun)


    mcts_params = DPWParams()
    mcts_params.d = d
    mcts_params.ec = 100
    mcts_params.n = n
    mcts_params.k = 0.5
    mcts_params.alpha = 0.85
    mcts_params.kp = 1.0
    mcts_params.alphap = 0.0
    mcts_params.clear_nodes = true
    mcts_params.maxtime_s = realmax(Float64)
    mcts_params.top_k = 1
    mcts_params.rng_seed = UInt64(0)
    result = stress_test(ast, mcts_params)
    # play_sequence(ast, result.action_seqs[1])
    return sim, result.rewards, result.action_seqs, result.r_history

end

carcolors = Dict{Int,Colorant}()
carcolors[1] = colorant"red"
carcolors[2] = colorant"blue"

# mean_rewards = []
# iterations = [250,500,1000,2000]
# for iter in iterations
# 	println("iter = ", iter)
#   rewards = []
# 	for seed in 1:10
tic()
sim, reward, action_seq, r_history = run_sim(100,[.01,.1],100,2000,18)
toc()

writedlm("scen120_rewards.txt", r_history)
    # reward = reward[1]
	# 	append!(rewards,reward)
frames = Frames(MIME("image/png"), fps=10)
for frame in 1 : length(sim.states)
    s = render(sim.states[frame], sim.roadway, cam=FitToContentCamera(), car_colors=carcolors)
    push!(frames, s)
end
println("creating gif...")
write(@sprintf("stress_test_R_%f.gif",reward[1]), frames)
	# end
	# append!(mean_rewards, mean(rewards))
# end

# writedlm("scen11_mean_rewards.txt", mean_rewards)
