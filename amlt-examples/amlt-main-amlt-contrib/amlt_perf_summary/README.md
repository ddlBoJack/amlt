# amlt-perf-summary

This plugin runs a function that computes a sequence metrics for selected or all jobs in the current project. The result is printed out as a table and can be optionally saved in a json file.

For example,
```
amlt perf-summary path/to/perf_module.py
```
may display
```
EXPERIMENT    MODIFIED     JOB               LAST_VALID_LOSS
------------  -----------  ----------------  -----------------
awesome-cat   a month ago  my-awesome-job-1            -6.4530
awesome-cat   a month ago  my-awesome-job-2            -6.6228
```

We expect the user to provide a module (`perf_module.py` in the above example) that implements `list_metrics` function. Each object in the list returned by `list_metrics` function should have a `name` attribute and a `get_value` method. For example, they can be defined as follows:

```python
import json
import os
import amlt_perf_summary.cli


def list_metrics():
    return [LastValidLoss()]


class LastValidLoss:
    name = "last valid loss"

    @staticmethod
    def get_value(amulet_job: "amlt_perf_summary.cli.AmuletJob"):
        """ Computes the summary metric. """
        results_dir = amulet_job.results(include=["loss.json"])
        with open(os.path.join(results_dir, "loss.json")) as f:
            data = json.load(f)
        return data["valid_loss"]
```

Here note that the client code can use `AmuletJob.results` method to download the results.
