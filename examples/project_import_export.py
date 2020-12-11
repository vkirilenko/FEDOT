from cases.data.data_utils import get_scoring_case_data_paths
from fedot.core.models.data import InputData
from fedot.core.composer.gp_composer.fixed_structure_composer import FixedStructureComposerBuilder
from fedot.core.composer.gp_composer.gp_composer import GPComposerRequirements
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, MetricsRepository
from fedot.core.repository.tasks import Task, TaskTypesEnum
from test.chain.test_chain_tuning import get_class_chain
from fedot.utilities.project_import_export import export_project_to_zip
from sklearn.metrics import roc_auc_score as roc_auc
from fedot.utilities.project_import_export import import_project_from_zip


def get_chain_by_composer(train_data_):
    available_model_types = ['logit', 'lda', 'knn']

    metric_function = MetricsRepository().metric_by_id(ClassificationMetricsEnum.ROCAUC)

    req = GPComposerRequirements(primary=available_model_types, secondary=available_model_types,
                                 pop_size=2, num_of_generations=1,
                                 crossover_prob=0.4, mutation_prob=0.5,
                                 add_single_model_chains=False)

    reference_chain = get_class_chain()
    builder = FixedStructureComposerBuilder(task=Task(TaskTypesEnum.classification)).with_initial_chain(
        reference_chain).with_metrics(metric_function).with_requirements(req)
    composer = builder.build()

    chain_ = composer.compose_chain(data=train_data_)

    return chain_


if __name__ == '__main__':
    train_file_path, test_file_path = get_scoring_case_data_paths()
    train_data = InputData.from_csv(train_file_path)
    test_data = InputData.from_csv(test_file_path)

    chain = get_chain_by_composer(train_data)

    # Export project to zipfile to directory '/home/user/Fedot/projects/project_name.zip'
    export_project_to_zip(chain, train_data, test_data, 'project_name.zip', 'project_name_log.log', verbose=True)
    # Import project from zipfile to Chain and InputData objects.
    chain, train_data, test_data = import_project_from_zip('/home/user/downloads/project_name.zip', verbose=True)

    chain.fit(train_data)
    prediction = chain.predict(test_data)

    print(roc_auc(test_data.target, prediction.predict))


