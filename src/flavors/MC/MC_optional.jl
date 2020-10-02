"""
    global_move(mc::MC, m::Model, conf) -> accepted::Bool

Propose a global move for configuration `conf`.
Returns wether the global move has been accepted or not.
"""
global_move(mc::MC, m::Model, conf) = false


"""
    init!(mc::MC, model)

Initializes a classical MonteCarlo model.
"""
init!(mc::MonteCarloFlavor, m::Model) = nothing