#!/bin/bash



# julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=1 --radius=24

# julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=31 --radius=24

# julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=101 --radius=24

# julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=501 --radius=24 
julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=1 --radius=24 --max_delay_agent=101 --nb_agent_delay=2 --runall=0

julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=1 --radius=24 --max_delay_agent=101 --nb_agent_delay=5 --runall=0

julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=1 --radius=24 --max_delay_agent=101 --nb_agent_delay=10 --runall=0

julia main-decentralized.jl --data_name='fashionmnist' --nb_agents=30 --batch_size=2 --nb_classes=10 --max_delay=1 --radius=24 --max_delay_agent=101 --nb_agent_delay=20 --runall=0