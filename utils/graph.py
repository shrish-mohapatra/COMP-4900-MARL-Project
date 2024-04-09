"""
Utilities for creating graphs from experiment results

end goals
ex. "LSTMMlp vs BaselineMLP"
goal="plot lstmmlp performance compared with baseline across all tasks"
x=iterations
y=mean_reward
plots = [
    SelectCriteria(model_config="LSTMMlp"),
    SelectCriteria(model_config="Mlp"),
]

ex. "decoder vs baseline"
goal="plot MappoDecoder performance compared with baseline Mappo across all tasks"
x=iterations
y=mean_reward
plots = [
    SelectCriteria(model_config="MappoDecoder"),
    SelectCriteria(model_config="Mappo"),
]

steps
1. dev provided config(
    csv_file_name: str file to extract data from (ex. eval_civilian_reward_episode_reward_mean.csv),
    exclude_folder: [str] = [] list of folders to exclude
) 
2. using config, load csv files matching csv_file_name
3. determine experiment metadata from csv file path
    - check folder time stamp to determine experiment order?
    example of result from step
    [
        {
            file_path: "...",
            content: pandas.DataFrame,
            experiment: {
                'task': task,
                'algorithm_config': algorithm_config,
                'seed': seed,
                'model_config': model_config,
                'critic_model_config': self.critic_model_config,
                'config': self.experiment_config,
            }
        }
    ]
4. using dev provided selection criteria, aggregate data
5. plot aggregated data
"""
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

from dataclasses import dataclass
from typing import List


@dataclass
class DataSource:
    file_path: str
    config: dict
    content: pd.DataFrame

    @classmethod
    def from_filepath(cls, file_path: str, config: dict):
        content = pd.read_csv(file_path, header=None)
        content.columns = ["x", "y"]
        return cls(file_path, config, content)


class SelectCriteria:
    """Criteria to aggregate experiment results"""

    def __init__(self, **args) -> None:
        """        
        Provide keyword arguments to apply filtering
        For example: SelectCriteria(model_config="MappoDecoder")
        """
        self.criteria = args

    def check(self, data_source: DataSource) -> bool:
        for config_key, config_val in self.criteria.items():
            if data_source.config[config_key] != config_val:
                return False
        return True

    def get_label(self) -> str:
        criteria_vals = iter(self.criteria.values())
        val1 = next(criteria_vals)
        val2 = next(criteria_vals)
        return f"{val1} x {val2}"


class GraphFactory:
    """Module to load and generate graphs from experiment results"""

    def __init__(
        self,
        experiment_config_file: str,
        source_csv_file: str,
        source_folder: str = "results",
        exclude_folders: List[str] = [],
        save_graphs = False,
        save_folder = "graphs",
    ):
        """
        Args
        - experiment_config_file: File where benchmark experiments JSON config is stored
        - source_csv_file: File to extract data from (ex. eval_civilian_reward_episode_reward_mean.csv)
        - soruce_filder: Folder where experiment results are saved
        - exclude_folders: List of folders to exclude from extraction
        - save_graphs: If yes will save PNGs to `save_graphs` folder
        - save_folder: Folder to save graphs to
        """
        self.experiment_config_file = experiment_config_file
        self.source_csv_file = source_csv_file
        self.source_folder = source_folder
        self.exclude_folders = exclude_folders
        self.save_graphs = save_graphs
        self.save_folder = save_folder

        self.data_sources: List[DataSource] = []
        self.experiment_configs: List[dict] = []

    def load(self):
        """Load experiment data based on configuration"""

        # Load experiment configs
        with open(self.experiment_config_file, 'r') as file:
            self.experiment_configs = json.load(file)

        # Load csv files
        file_paths = []
        for dirpath, _, files in os.walk(self.source_folder):
            # Check for excluded folders
            exclusion_check = [
                excluded_folder
                for excluded_folder in self.exclude_folders
                if excluded_folder in dirpath
            ]
            if exclusion_check:
                continue

            # Add files
            for file in files:
                if file == self.source_csv_file:
                    file_path = os.path.join(dirpath, file)
                    creation_time = os.path.getctime(dirpath)

                    file_paths.append({
                        "file_path": file_path,
                        "creation_time": creation_time,
                    })

        # Determine file order
        # using creation time to map to experiment config
        # this is pretty sketch.
        file_paths.sort(key=lambda x: x["creation_time"])
        self.data_sources = [
            DataSource.from_filepath(
                file_path=file_paths[i]["file_path"],
                config=self.experiment_configs[i],
            )
            for i in range(len(file_paths))
        ]

    def graph(
        self,
        title: str,
        criterias: List[SelectCriteria],
        x_label="Iteration",
        y_label="Mean Return",
    ):
        """
        plt.clf()

        for each criteria
            dataframes = []
            for each datasource
                for each key,val in criteria
                    if datasource.config[key] == value:
                        dataframes.append(datasource.df)

            calc aggregated df
                calc avg x across same y vals

            scatter aggregated df
            plot line aggregated df
        
        dfs = [
            [x: [1, 2, 3], y: [1, 2, 3]],
            [x: [1, 2, 3, 4], y: [3, 4, 5, 6]],
        ]

        avg_dfs = [
            [x: [1, 2, 3, 4], y: [2, 3, 4, 6]],
        ]
        """
        # plt.clf()
        # _, ax = plt.subplots()

        # Prepare dataframes
        for criteria in criterias:
            dfs: List[pd.DataFrame] = []
            for data_source in self.data_sources:
                if criteria.check(data_source):
                    dfs.append(data_source.content)

            df_concat = pd.concat(dfs)
            df_avg = df_concat.groupby('x', as_index=False).mean()
            
            plt.scatter(df_avg["x"], df_avg["y"], label=str(criteria.get_label()))
            plt.plot(df_avg["x"], df_avg["y"], label=str(criteria.get_label()))
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        
        if self.save_graphs:
            graph_file_path = f"{self.save_folder}/{title}.png"
            plt.savefig(graph_file_path)
            print("Saved graph to", graph_file_path)
        else:
            plt.show()
        
        plt.clf()


def test_checkpoint():
    checkpoint_path = "results\mappo_simple_speaker_listener_lstmmlp__3c667ef9_24_04_07-22_51_09\checkpoints\checkpoint_300000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print(checkpoint["loss_listener"].keys())


if __name__ == "__main__":
    gf = GraphFactory(
        experiment_config_file="experiment.config",
        source_csv_file="eval_listener_reward_episode_reward_mean.csv",
        # source_csv_file="collection_listener_reward_episode_reward_mean.csv",
        exclude_folders=["_saved"],
        save_graphs=False,
    )
    gf.load()
    
    """
    3 algos (mappo, mappodecoder, curriculum)
    2 models (mlp, lstmmlp)
    graph per Task
    - baseline
        - algo: mappo model: mlp
        - algo: mappodecoder model: mlp
        - algo: mappo model: lstmmlp
        - algo: mappodecoder model: lstmmlp
        - algo: curriculum
    
    -e 
    """

    gf.graph(
        "Baseline Task",
        criterias=[
            SelectCriteria(
                algorithm_config="MappoConfig",
                model_config="MlpConfig",
                task="BaselineVmasTask",
            ),
            SelectCriteria(
                algorithm_config="MappoConfig",
                model_config="LSTMMlpConfig",
                task="BaselineVmasTask",
            ),
            SelectCriteria(
                algorithm_config="MappoDecoderConfig",
                model_config="MlpConfig",
                task="BaselineVmasTask",
            ),
            SelectCriteria(
                algorithm_config="MappoDecoderConfig",
                model_config="LSTMMlpConfig",
                task="BaselineVmasTask",
            ),
            SelectCriteria(
                algorithm_config="MappoCurriculum",
                model_config="MlpConfig",
                task="BaselineVmasTask",
            ),
        ]
    )

    # gf.graph(
    #     "Task Peformance Comparison",
    #     criterias=[
    #         SelectCriteria(task="BaselineVmasTask"),
    #         SelectCriteria(task="Ext1VmasTask"),
    #         SelectCriteria(task="Ext2VmasTask"),
    #         SelectCriteria(task="Ext3VmasTask"),
    #         SelectCriteria(task="Ext4VmasTask"),
    #     ],
    # )
    # gf.graph(
    #     "Mlp vs LSTMMlp",
    #     criterias=[
    #         SelectCriteria(model_config="MlpConfig"),
    #         SelectCriteria(model_config="LSTMMlpConfig"),
    #     ],
    # )
    # gf.graph(
    #     "Curriculum vs Baseline",
    #     criterias=[
    #         SelectCriteria(algorithm_config="MappoCurriculum", task="BaselineVmasTask"),
    #         SelectCriteria(algorithm_config="MappoConfig", task="BaselineVmasTask"),
    #     ],
    # )

    # test_checkpoint()
    pass
