from dataclasses import dataclass
import os
from typing import List, Optional

import click
import amlt
from amlt.cli.base import pass_project, ExpAndJobsParam, ExpSubParam


@dataclass(frozen=True)
class AmuletJob:
  """
  Easy access to all properties of a single Amulet job (project name, experiment name, config, environment, code, and results).
  """
  output_dir: str
  tempdir: str
  jobs: "amlt.api.jobs.JobsClient"
  update_results: bool

  @property
  def project_name(self) -> str:
    return self.jobs._project.name

  @property
  def experiment_name(self) -> str:
    return self.jobs._experiment.name

  @property
  def job_name(self) -> str:
    return self.jobs.config.name

  @property
  def config(self):
    return self.jobs.config

  @property
  def canonical_jobdir(self) -> str:
    return os.path.join(
      self.output_dir,
      self.experiment_name,
      self.jobs.config.local_directory_name,
    )

  @property
  def temp_jobdir(self) -> str:
    return os.path.join(
      self.tempdir,
      self.jobs._experiment.name,
      self.jobs.config.local_directory_name,
    )

  def pull_results(self, include: Optional[List[str]] = None, update: bool = False) -> str:
    """
    Actually pulls the results.
    Args:
      include: (optional) list of patterns of files (e.g., "*.yaml").
      update: if True, download into amulet's output_dir.
    Returns:
      the directory name containing the results (self.canonical_jobdir or self.temp_jobdir).
    """
    download_dir = self.output_dir if update else self.tempdir
    self.jobs.results.pull(download_dir, include=include, show_progress=True)
    return self.canonical_jobdir if update else self.temp_jobdir

  def results(self, include: Optional[List[str]] = None) -> str:
    """
    Pulls results if necessary and returns the name of the directory containing the results.
    Args:
      include: (optional) list of patterns of files (e.g., "*.yaml").
    Returns:
      the directory name containing the results (self.canonical_jobdir or self.temp_jobdir).
    """
    import glob
    from amlt.helpers import LOG
    if self.update_results:
      # unconditionally download into canonical_jobdir
      return self.pull_results(include=include, update=True)
    for jobdir in [self.canonical_jobdir, self.temp_jobdir]:
      if not os.path.isdir(jobdir):
        continue
      if include is None and len(os.listdir(jobdir)) > 0:
        return jobdir
      existing = True
      for inc in include:
        files = glob.glob(os.path.join(jobdir, "**", inc), recursive=True)
        LOG.info(f"Checking if {jobdir} contains {inc}... found {files}")
        existing &= len(files) > 0
      if existing:
        return jobdir
    return self.pull_results(include=include)

  def __repr__(self):
    return (
      f"AmuletJob(project={self.project_name}, experiment={self.experiment_name},"
      f" job={self.job_name}, canonical_jobdir={self.canonical_jobdir})"
    )


def exp_job_arg_decorator(cmd):
  exps_and_jobs_param = ExpAndJobsParam(
    ExpSubParam("experiments_and_jobs", default_exp_ok=True, load=True, exp_none_ok=True, n_jobs=-1)
  )

  cmd = exps_and_jobs_param.validate(cmd)
  cmd = pass_project('project', 'output_path')(cmd)
  cmd = exps_and_jobs_param.register(cmd)
  return cmd


def import_module_from_path(path):
  """ Imports module from path """
  import importlib.util
  import os
  # These three lines are for loading a module from a file in Python 3.5+
  # https://bugs.python.org/issue21436
  module_name = os.path.basename(path).split(".")[0]
  spec = importlib.util.spec_from_file_location(module_name, path)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def from_project_or_experiments_and_jobs(experiments_and_jobs, project):
  """ Generator that yields JobsClient for selected jobs, experiments, or a project. """
  if experiments_and_jobs.jobs is not None:
    for job in experiments_and_jobs.jobs:
      yield job._experiment, job
  elif experiments_and_jobs.exp is not None:
    experiment = experiments_and_jobs.exp
    for job in experiment.jobs:
      yield experiment, job
  else:
    for exp in project.experiments:
      for job in exp.jobs:
        yield exp, job


@click.command("perf-summary")
@click.argument("perf-module-path", type=str)
@click.option("--json-output", type=str, help="(optional) path to write output to a file in json format.")
@click.option("--update-results", is_flag=True, help="Update the results if they exist in output directory.")
@exp_job_arg_decorator
def perf_summary(
  experiments_and_jobs,
  project,
  output_path,
  perf_module_path,
  json_output,
  update_results,
):
  """
  Summarize performance for all or selected jobs.

  PERF_MODULE_PATH (e.g., path/to/module.py) is the path to the module that implements the list_metrics function.
  """
  # local import to speed up tab completion
  import json
  from tempfile import TemporaryDirectory
  from amlt.helpers.table import TableOutput, TimeCell
  perf_mod = import_module_from_path(perf_module_path)
  metrics = perf_mod.list_metrics()
  table = TableOutput(["experiment", "modified", "job"] + [m.name for m in metrics])
  with TemporaryDirectory() as tempdir:
    for exp, job in from_project_or_experiments_and_jobs(experiments_and_jobs, project):
      amulet_job = AmuletJob(output_path, tempdir, job, update_results)
      metric_values = [m.get_value(amulet_job) for m in metrics]
      table.add_row((exp.name, TimeCell(exp.model.modified_at), job.config.name) + tuple(metric_values))
  table.sort(key="modified")
  print(table)
  if json_output:
    with open(json_output, "wt") as f:
      json.dump(table.table_to_json(), f)
