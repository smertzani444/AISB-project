#!/usr/bin/env python3
#!/usr/bin/env python3
import os
import sys
import urllib.request
import tarfile
import requests
import urllib3

def main():
    # 1) Base folder (change to your project root if you like)
    PEPRMINT_FOLDER = os.getcwd()

    # 2) Sub-folders
    WORKDIR         = os.path.join(PEPRMINT_FOLDER, "dataset")
    CATHFOLDER      = os.path.join(PEPRMINT_FOLDER, "databases", "cath")
    ALPHAFOLDFOLDER = os.path.join(PEPRMINT_FOLDER, "databases", "alphafold")
    PROSITEFOLDER   = os.path.join(PEPRMINT_FOLDER, "databases", "prosite")
    UNIPROTFOLDER   = os.path.join(PEPRMINT_FOLDER, "databases", "uniprot")
    FIGURESFOLDER   = os.path.join(PEPRMINT_FOLDER, "figures")

    ALL_FOLDERS = [
        WORKDIR, CATHFOLDER, ALPHAFOLDFOLDER,
        PROSITEFOLDER, UNIPROTFOLDER, FIGURESFOLDER
    ]

    # 3) Create them
    for d in ALL_FOLDERS:
        os.makedirs(d, exist_ok=True)
    print(f"✅ Created folder structure under {PEPRMINT_FOLDER}")

    # 4) Download PROSITE alignments
    PROSITE_URL   = "ftp://ftp.expasy.org/databases/prosite/prosite_alignments.tar.gz"

    archive_path  = os.path.join(PROSITEFOLDER, "prosite_alignments.tar.gz")  
    if not os.path.exists(archive_path):
        urllib.request.urlretrieve(PROSITE_URL, archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=PROSITEFOLDER)

    print(f"↓ Downloading PROSITE alignments to {archive_path}")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    urllib.request.urlretrieve(PROSITE_URL, archive_path)
    print("→ Extracting…")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=PROSITEFOLDER)
    print(f"✅ PROSITE data ready in {PROSITEFOLDER}")

    # 5) Download CATH domain list
    CATH_VERSION = "v4_2_0"
    DOMAIN_LIST_URL = (
        f"http://download.cathdb.info/"
        f"cath/releases/all-releases/{CATH_VERSION}/"
        f"cath-classification-data/cath-domain-list-{CATH_VERSION}.txt"
    )
    out_path = os.path.join(CATHFOLDER, f"cath-domain-list-{CATH_VERSION}.txt")
    print(f"↓ Fetching CATH domain list ({CATH_VERSION}) to {out_path}")

    with requests.get(DOMAIN_LIST_URL, stream=True, timeout=60, verify=False) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    fh.write(chunk)

    print(f"✅ CATH domain list saved to {out_path}")

if __name__ == "__main__":
    main()