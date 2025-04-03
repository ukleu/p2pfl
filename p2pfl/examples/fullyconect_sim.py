import argparse
import time

import matplotlib.pyplot as plt

from p2pfl.communication.protocols.grpc.grpc_communication_protocol import GrpcCommunicationProtocol
from p2pfl.communication.protocols.memory.memory_communication_protocol import InMemoryCommunicationProtocol
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import wait_convergence, wait_to_finish


def set_standalone_settings(disable_ray: bool = False) -> None:
    Settings.GRPC_TIMEOUT = 0.5
    Settings.HEARTBEAT_PERIOD = 5
    Settings.HEARTBEAT_TIMEOUT = 40
    Settings.GOSSIP_PERIOD = 1
    Settings.TTL = 40
    Settings.GOSSIP_MESSAGES_PER_PERIOD = 100
    Settings.AMOUNT_LAST_MESSAGES_SAVED = 100
    Settings.GOSSIP_MODELS_PERIOD = 1
    Settings.GOSSIP_MODELS_PER_ROUND = 4
    Settings.GOSSIP_EXIT_ON_X_EQUAL_ROUNDS = 10
    Settings.TRAIN_SET_SIZE = 4
    Settings.VOTE_TIMEOUT = 60
    Settings.AGGREGATION_TIMEOUT = 60
    Settings.WAIT_HEARTBEATS_CONVERGENCE = 0.2 * Settings.HEARTBEAT_TIMEOUT
    Settings.LOG_LEVEL = "INFO"
    Settings.EXCLUDE_BEAT_LOGS = True
    Settings.DISABLE_RAY = disable_ray
    logger.set_level(Settings.LOG_LEVEL)

def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment using the File Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=10)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=2)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--protocol", type=str, help="The protocol to use.", default="memory", choices=["grpc", "unix", "memory"])
    parser.add_argument("--framework", type=str, help="The framework to use.", default="pytorch", choices=["pytorch", "tensorflow", "flax"])
    parser.add_argument("--aggregator", type=str, help="The aggregator to use.", default="fedavg", choices=["fedavg", "scaffold"])
    parser.add_argument("--reduced_dataset", action="store_true", help="Use a reduced dataset just for testing.", default=True)
    parser.add_argument("--use_scaffold", action="store_true", help="Use the Scaffold aggregator.", default=False)
    parser.add_argument("--disable_ray", action="store_true", help="Disable Ray.", default=False)
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="full",
        help="The network topology (star, full, line, ring).",
    )
    args = parser.parse_args()
    
    args.topology = TopologyType(args.topology)

    return args

def create_tensorflow_model() -> P2PFLModel:
    """Create a TensorFlow model."""
    import tensorflow as tf  # type: ignore

    from p2pfl.learning.frameworks.tensorflow.keras_model import MLP as MLP_KERAS
    from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel

    model = MLP_KERAS()  
    model(tf.zeros((1, 28, 28, 1)))
    return KerasModel(model)

def create_pytorch_model() -> P2PFLModel:
    """Create a PyTorch model."""
    from p2pfl.learning.frameworks.pytorch.lightning_model import MLP, LightningModel

    return LightningModel(MLP())  

def mnist(
    n: int,
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "memory",
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = True,
    topology: TopologyType = TopologyType.FULL,
) -> None:

    if measure_time:
        start_time = time.time()

    if n > Settings.TTL:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes." "Otherwise, some messages will not be delivered."
        )
    
    if framework == "tensorflow":
        from p2pfl.learning.frameworks.tensorflow.keras_learner import KerasLearner

        model_fn = create_tensorflow_model
        learner = KerasLearner
    elif framework == "pytorch":
        from p2pfl.learning.frameworks.pytorch.lightning_learner import LightningLearner

        model_fn = create_pytorch_model
        learner = LightningLearner  
    else:
        raise ValueError(f"Framework {args.framework} not added on this example.")
    
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    partitions = data.generate_partitions(
        n * 50 if reduced_dataset else n,
        RandomIIDPartitionStrategy, 
    )
    
    nodes = []
    for i in range(n):
        address = f"node-{i}" if protocol == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol == "unix" else "127.0.0.1"

        
        node = Node(
            model_fn(),
            partitions[i],
            learner=learner,
            protocol=InMemoryCommunicationProtocol if protocol == "memory" else GrpcCommunicationProtocol,
            address=address,
            simulation=True,
            aggregator=Scaffold() if aggregator == "scaffold" else None,
        )
        node.start()
        nodes.append(node)
    
    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        wait_convergence(nodes, n - 1, only_direct=False, wait=60)  

        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

       
        nodes[0].set_start_learning(rounds=r, epochs=e)

        
        wait_to_finish(nodes, timeout=60 * 60)  

        
        if show_metrics:
            local_logs = logger.get_local_logs()
            if local_logs != {}:
                logs_l = list(local_logs.items())[0][1]
                
                for round_num, round_metrics in logs_l.items():
                    for node_name, node_metrics in round_metrics.items():
                        for metric, values in node_metrics.items():
                            x, y = zip(*values)
                            plt.plot(x, y, label=metric)
                            
                            plt.scatter(x[-1], y[-1], color="red")
                            plt.title(f"Round {round_num} - {node_name}")
                            plt.xlabel("Epoch")
                            plt.ylabel(metric)
                            plt.legend()
                            plt.show()

            
            global_logs = logger.get_global_logs()
            if global_logs != {}:
                logs_g = list(global_logs.items())[0][1]  
                
                for node_name, node_metrics in logs_g.items():
                    for metric, values in node_metrics.items():
                        x, y = zip(*values)
                        plt.plot(x, y, label=metric)
                        
                        plt.scatter(x[-1], y[-1], color="red")
                        plt.title(f"{node_name} - {metric}")
                        plt.xlabel("Epoch")
                        plt.ylabel(metric)
                        plt.legend()
                        plt.show()
    except Exception as e:
        raise e
    finally:
        
        for node in nodes:
            node.stop()

        if measure_time:
            print("--- %s seconds ---" % (time.time() - start_time))

    if __name__ == "__main__":
        
        args = __parse_args()

        set_standalone_settings(disable_ray=args.disable_ray)

        #set logger check wether correct? import file logger? import decorator?
        logger.setup_file_handler()
        # might require try block
        mnist(
            args.nodes,
            args.rounds,
            args.epochs,
            show_metrics=args.show_metrics,
            measure_time=args.measure_time,
            protocol=args.protocol,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=args.reduced_dataset,
            topology=args.topology,
        )
