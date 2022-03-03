import click
from amlt.globals import LOG
from amlt.cli.base import target_config_option, exp_arg_job_opt


@click.command("cleanup")
@click.option(
  "-I",
  "--include",
  multiple=True,
  metavar="FILENAME/PATTERN",
  required=True,
  help="Delete files matching this pattern",
)
@exp_arg_job_opt(load_experiments=True, n_experiments=1)
def cleanup(project, experiment_and_jobs, include):
  """
  Remove files matching a specific pattern from a job
  """
  jobs = (
    experiment_and_jobs.jobs
    if experiment_and_jobs.jobs
    else experiment_and_jobs.exp.jobs
  )
  for job in jobs:
    LOG.debug(f"Processing {job.config.name}")
    transport = job.results.storage.transport
    for blob in transport.list_directory(
      job.config.results_dir,
      include=include,
      include_metadata=True,
      recursive=True
    ):
      LOG.info("deleting %s", blob.name)
      transport.delete_blob(blob.name)


@click.command("cleanup-job")
@target_config_option
@click.option("--vc", help="Virtual cluster")
@click.option(
  "--pre",
  "--preemptible",
  "preemptible",
  is_flag=True,
  help="Indicate that job may be preempted.",
)
@click.option(
  "-I",
  "--include",
  multiple=True,
  metavar="FILENAME/PATTERN",
  required=True,
  help="Delete files matching this pattern",
)
@exp_arg_job_opt(load_experiments=True, n_experiments=1, n_jobs=1)
def cleanup_job(project, experiment_and_job, include, target_override, vc, preemptible):
  """
  Remove files matching a specific pattern from a job by running a deletion job on the server.
  """
  from amlt.api.project import ProjectClient
  from amlt.config.core import (
    AMLTConfig,
    EnvironmentConfig,
    JobCommandConfig,
    TargetConfig,
  )
  from amlt.api.run import ConfigRunClient
  from amlt.exceptions import ArgumentException

  if target_override is None:
    raise ArgumentException("Please provide a target to run on.")

  project: ProjectClient = project  # for IDE completion

  job = experiment_and_job.jobs[0]

  # create a new experiment using the same storage as exp
  cleanup_exp = project.experiments.create()
  config = AMLTConfig(
    target=project.targets.find(target_override, vc=vc),
    environment=EnvironmentConfig(image="pytorch/pytorch:latest"),
    storage=job.config.storage,
    code=None,
  )
  if preemptible:
    config.target.queue = TargetConfig.PREEMPTIBLE_QUEUE

  # build the command to run using job results paths and supplied patterns
  mount_dir = job.results.storage.storage_config.mount_dir
  command = []
  for inc in include:
    command.append("rm -r {}/{}/{}".format(mount_dir, job.config.results_dir, inc))

  # run the clean-up job
  job = JobCommandConfig("cleanup", command)
  run_client = ConfigRunClient(
    cleanup_exp, config=config, run_options=ConfigRunClient.RunOptions()
  )
  jobs_client = run_client.run([job])
  (jobs_status_table, _), _ = jobs_client.status.get_tables()
  print(jobs_status_table)
