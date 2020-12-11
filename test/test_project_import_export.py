import numpy as np
import pytest
import os
import shutil
import zipfile

from sklearn.datasets import load_iris

from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.models.data import InputData
from fedot.core.utils import default_fedot_data_dir
from fedot.utilities.project_import_export import export_project_to_zip, import_project_from_zip
from test.test_chain_import_export import create_chain

DEFAULT_PROJECTS_PATH = os.path.join(default_fedot_data_dir(), 'projects')
PATHS_TO_DELETE_AFTER_TEST = []


@pytest.fixture(scope="session", autouse=True)
def creation_model_files_before_after_tests(request):
    request.addfinalizer(delete_files_folders)


def delete_files_folders():
    for path_to_del in PATHS_TO_DELETE_AFTER_TEST:
        absolute_path = os.path.join(DEFAULT_PROJECTS_PATH, path_to_del)
        if os.path.isfile(absolute_path):
            os.remove(absolute_path)
        if os.path.isdir(absolute_path):
            shutil.rmtree(absolute_path)


def data_setup():
    predictors, response = load_iris(return_X_y=True)
    np.random.shuffle(predictors)
    np.random.shuffle(response)
    predictors = predictors[:100]
    response = response[:100]
    data = InputData(features=predictors, target=response, idx=np.arange(0, 100),
                     task=Task(TaskTypesEnum.classification),
                     data_type=DataTypesEnum.table)
    return data


def test_export_project_correctly():
    folder_name = 'iris_classification'
    zip_name = folder_name + '.zip'
    name_of_log = 'log.log'
    path_to_zip = os.path.join(DEFAULT_PROJECTS_PATH, zip_name)

    PATHS_TO_DELETE_AFTER_TEST.append(zip_name)

    chain = create_chain()
    data = data_setup()
    export_project_to_zip(chain, data, folder_name, log_file_name=name_of_log, verbose=True)

    assert os.path.exists(path_to_zip)

    with zipfile.ZipFile(path_to_zip) as zip_object:
        assert sorted([file.filename for file in zip_object.infolist()]) == sorted(
            ['log.log', 'data.csv', 'chain.json'])


def test_import_project_correctly():
    folder_name = 'iris_classification'
    zip_name = folder_name + '.zip'
    zip_path = os.path.join(DEFAULT_PROJECTS_PATH, zip_name)
    folder_path = os.path.join(DEFAULT_PROJECTS_PATH, folder_name)

    PATHS_TO_DELETE_AFTER_TEST.append(folder_name)

    import_project_from_zip(zip_path)

    assert os.path.exists(folder_path)
    assert sorted([file for file in os.listdir(folder_path)]) == sorted(['log.log', 'data.csv', 'chain.json'])
