module WflowRoutingGNN

using Printf
using Statistics

include("preprocess.jl")
export ldd_to_graph, build_wflow_graph, make_horizon_dataset, DOMAIN_VARS, LDD_OFFSETS,
       check_and_correct_grid_alignment,
       scale_river_q!, scale_river_h!, VAR_SCALERS,
       DataSettings, save_data_settings, load_data_settings,
       get_timestep

include("gnn.jl")
export WflowGNN, MassBalanceLayer, mb_diagnostics, ModelSettings, save_model_settings, load_model_settings, ACTIVATIONS,
       SparseConv

include("strategy.jl")
export TrainingStrategy, save_training_strategy, load_training_strategy,
       update_steps!, loss_function, one_step_loss

include("training.jl")
export TrainSettings, save_train_settings, load_train_settings, train_model!

include("run.jl")
export run_wflow_gnn, run_wflow_gnn_from_toml

include("hparsearch.jl")
export HParSearchSettings, save_hpar_search_settings, load_hpar_search_settings, hpar_search

include("plot.jl")
export plot_losses, plot_validation_movie, plot_timeseries, plot_downstream_timeseries,
       plot_mb_diagnostics

include("rollout.jl")
export rollout, evaluate_trajectory, rollout_mb_diagnostics

include("postprocess.jl")
export regrid, write_regrid_to_netcdf

end # module WflowRoutingGNN
