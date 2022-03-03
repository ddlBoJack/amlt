import click
from amlt.cli.base import pass_project
from amlt.helpers.table import TableOutput


@click.command("job-id-map")
@pass_project("project")
def job_id_map(project):
  """
  print a mapping for each job id -> exp name / job name
  """
  table = TableOutput("exp job path".split())
  for exp in project.experiments:
    for job in exp.jobs:
      path = job.config.results_dir
      table.add_row((exp.name, job.config.name, path))
  print(table)
