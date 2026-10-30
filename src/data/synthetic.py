# One function / class 
# IN: That take in the task name, distance, number of graphs,seed, filler noise etc
# DOES: It calls src/graphs, draws one label per graph  and build each Data with the right attribute: x, edge index,y, target_mask etc...
# Metadata to attach: source index, target index and true distance. The metrics need them, and the tests check them.