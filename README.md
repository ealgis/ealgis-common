# Development

Add `./ealgis-common:/ealgis-common` to your `volumes` for `web` in your ealgis `docker-compose.yml` file.

Run inside your ealgis_web_1 container:

```
pip uninstall --yes ealgis_common && pip install --editable /ealgis-common
```

https://stackoverflow.com/a/38997026

Setup an automatic rsync from the root of your Ealgis directory (assuming your ealgis-common repo is in the same parent folder):

```
alias run_rsync='rsync -va --delete ../ealgis-common/ ./ealgis-common'
run_rsync; fswatch -o ../ealgis-common | while read f; do run_rsync; done
```
