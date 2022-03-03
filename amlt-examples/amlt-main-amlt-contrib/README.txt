Example on how to extend amlt.

Synopsis:

First install the plugin:
$ pip install .
$ export AMLT_ENABLE_EXTENSIONS=1

Then, the command is available through amlt:
$ amlt --help
$ amlt cleanup --help
$ amlt cleanup <experiment> [<job1>, ...] -I "*tfevents*"
$ amlt cleanup-job <experiment> [<job1>, ...] -I "*tfevents*"
$ amlt aml-runs --help


There are three commands:
* cleanup loops over all jobs and over all their results and deletes those matching the specification.
* cleanup-job does the same, but inside a job that it submits.
* aml-runs: prints the aml runs of the specified jobs (you'll want to put something more sensible here)


Why as a plugin, not a separate tool?
* The plugin gives you completion for experiments and job names
* No need to explicitly deal with authentication
