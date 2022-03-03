import click
from amlt.cli.base import exp_arg_job_opt


@click.command("aml-runs")
@exp_arg_job_opt(load_experiments=True, n_experiments=1)
def aml_runs(project, experiment_and_jobs):
  """
  obtain AML run objects for the given experiments/jobs
  """
  jobs = (
    experiment_and_jobs.jobs
    if experiment_and_jobs.jobs
    else experiment_and_jobs.exp.jobs
  )
  for job in jobs:
    target_client = job.config.target_client
    job_id = job.config.id
    print(target_client.get_run(job_id))
