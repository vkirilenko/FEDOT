import os
import shutil
from typing import List

from fedot.core.composer.chain import Chain
from fedot.core.models.data import InputData
from fedot.core.utils import default_fedot_data_dir
from fedot.core.log import default_log

DEFAULT_PATH = default_fedot_data_dir()
DEFAULT_PROJECTS_PATH = os.path.join(DEFAULT_PATH, 'projects')


def export_project_to_zip(chain: Chain, train_data: InputData, test_data: InputData, zip_name: str,
                          log_file_name: str = None, verbose: bool = False):
    # TODO add functionality for compress more files that user wants
    # TODO add functionality for compress fitted models
    """
    Convert chain to JSON, data to csv, compress them to zip
    archive and save to 'DEFAULT_PROJECTS_PATH/projects' with logs.

    :param chain: Chain object to export
    :param train_data: train InputData object to export
    :param test_data: test InputData object to export
    :param zip_name: name of the zip file
    :param log_file_name: name of the file with log to export
    :param verbose: flag to write logs
    """

    log = default_log('fedot.utilities.project_import_export')
    absolute_folder_path, absolute_zip_path, folder_name, zip_name = _prepare_paths(zip_name)
    _check_for_existing_project(absolute_folder_path, log, verbose)

    # Converts python objects to files for compression
    chain.save_chain(os.path.join(absolute_folder_path, 'chain.json'))
    train_data.to_csv(os.path.join(absolute_folder_path, 'train_data.csv'), header=True)
    test_data.to_csv(os.path.join(absolute_folder_path, 'test_data.csv'), header=True)
    _copy_log_file(log_file_name, absolute_folder_path, log, verbose)

    shutil.make_archive(absolute_folder_path, 'zip', absolute_folder_path)
    shutil.rmtree(absolute_folder_path)

    if verbose:
        log.info(f'The project was saved on the path: {absolute_folder_path}')


def import_project_from_zip(zip_path: str, verbose: bool = False) -> [Chain, InputData, InputData]:
    """
    Unzipping zip file. Zip file should contains:
    - chain.json: json performance,
    - train_data.csv: csv with first line which contains task_type and data_type of train InputData object.
    - test_data.csv: csv with first line which contains task_type and data_type of test InputData object.

    Created Chain and InputData objects. Ready to work with it.

    :param zip_path: path to zip archive
    :param verbose: flag to write logs
    :return [Chain, InputData, InputData]: return array of Chain and InputData objects.
    """
    log = default_log('fedot.utilities.project_import_export')
    train_data, test_data, chain = [None, None, None]

    _check_zip_path(zip_path, log)
    zip_name = _get_zip_name(zip_path)
    folder_path = os.path.join(DEFAULT_PROJECTS_PATH, zip_name)

    shutil.unpack_archive(zip_path, folder_path)

    if verbose:
        message = f"The project '{zip_name}' was unpacked to the '{folder_path}'."
        log.info(message)

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('json'):
                chain = Chain()
                chain.load_chain(os.path.join(root, file))
            if file == 'train_data.csv':
                train_data = InputData.from_csv(os.path.join(root, file), header=True)
            if file == 'test_data.csv':
                test_data = InputData.from_csv(os.path.join(root, file), header=True)

    for file, file_name in [[train_data, 'train data'], [test_data, 'test data'], [chain, 'JSON chain']]:
        _check_for_exist_file_raise(file, file_name, verbose, log)

    return [chain, train_data, test_data]


def _check_for_exist_file_raise(file: InputData or None or Chain, file_name: str, verbose: bool, log: 'Log'):
    if file is None:
        message = f"No {file_name} in the project folder."
        if verbose:
            log.error(message)
        raise ValueError(message)

def _get_zip_name(zip_path: str) -> str:
    zip_path_split = os.path.split(zip_path)
    zip_name = zip_path_split[-1].split('.')
    return zip_name[0]


def _check_zip_path(zip_path: str, log: 'Log'):
    """Check 'zip_path' for correctness."""

    if not os.path.exists(zip_path):
        message = f"File with the path '{zip_path}' could not be found."
        log.error(message)
        raise FileExistsError(message)

    zip_path_split = os.path.split(zip_path)

    if zip_path_split[-1].split('.')[-1] != 'zip':
        message = f"Zipfile must be with 'zip' extension."
        log.error(message)
        raise FileExistsError(message)


def _copy_log_file(log_file_name: str, absolute_folder_path: str, log: 'Log', verbose: bool):
    """Copy log file to folder which will be compressed."""

    if log_file_name is not None:
        if not os.path.isabs(log_file_name):
            log_file_name = os.path.abspath(os.path.join(DEFAULT_PATH, log_file_name))

        if os.path.exists(log_file_name):
            shutil.copy2(log_file_name, os.path.join(absolute_folder_path, os.path.split(log_file_name)[-1]))
        else:
            message = f"No log file with the name '{log_file_name}'."
            if verbose:
                log.error(message)
            raise FileExistsError(message)


def _prepare_paths(zip_name: str) -> List[str]:
    """Prepared absolute paths for zip and project's folder."""

    name_split = zip_name.split('.')
    folder_name = zip_name

    if len(name_split) == 2:
        folder_name = name_split[0]
    else:
        zip_name = zip_name + '.zip'

    absolute_folder_path = os.path.join(DEFAULT_PROJECTS_PATH, folder_name)
    absolute_zip_path = os.path.join(absolute_folder_path, zip_name)

    return [absolute_folder_path, absolute_zip_path, folder_name, zip_name]


def _check_for_existing_project(absolute_folder_path, log, verbose):
    """Check for existing folder and zipfile of project. Create it, if it is no exists."""

    if os.path.exists(absolute_folder_path + '.zip'):
        message = f"Zipfile with the name '{absolute_folder_path + '.zip'}' exists."
        if verbose:
            log.error(message)
        raise FileExistsError(message)

    if os.path.exists(absolute_folder_path):
        message = f"Project with the name '{absolute_folder_path}' exists."
        if verbose:
            log.error(message)
        raise FileExistsError(message)
    else:
        os.makedirs(absolute_folder_path)
