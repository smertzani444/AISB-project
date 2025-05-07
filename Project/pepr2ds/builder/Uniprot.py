#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MIT License
# 
# Copyright (c) 2022 Reuter Group
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


""" Module to fetch info from the Uniprot database while creating PePrMInt

Just imported the source file in Alexander's branch (that we're choosing not
to merge)

__author__ = "Alexander Popescu"
__organization__ = "Computational Biology Unit, Universitetet i Bergen"
__copyright__ = "Copyright (c) 2022 Reuter Group"
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Phillippe Samer"
__email__ = "samer@uib.no"
__status__ = "Prototype"
"""

##For more information visit https://www.uniprot.org/help/id_mapping

import re
import time
import json
import zlib
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs, urlencode
import requests
from requests.adapters import HTTPAdapter, Retry
from typing import Optional, List


API_URL = "https://rest.uniprot.org"
POLLING_INTERVAL = 3
MAX_TRIES_UPON_FAILED_REQUESTS = 5


retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))


def check_response(response):
    try:
        response.raise_for_status()
    except requests.HTTPError:
        print(response.json())
        raise


def submit_id_mapping(from_db, to_db, ids):
    attempt = 1
    while attempt <= MAX_TRIES_UPON_FAILED_REQUESTS:
        try:
            request = requests.post(
                f"{API_URL}/idmapping/run",
                data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
            )
            check_response(request)
            return request.json()["jobId"]
        except:
            if attempt < MAX_TRIES_UPON_FAILED_REQUESTS:
                print(f"Got code {request.status_code} (should be 'ok'=={requests.codes.ok})")
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
                ++attempt
            else:
                raise


def get_next_link(headers):
    re_next_link = re.compile(r'<(.+)>; rel="next"')
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def check_id_mapping_results_ready(job_id):
    while True:
        request = session.get(f"{API_URL}/idmapping/status/{job_id}")
        check_response(request)
        j = request.json()
        if "jobStatus" in j:
            if j["jobStatus"] == "RUNNING":
                print(f"Retrying in {POLLING_INTERVAL}s")
                time.sleep(POLLING_INTERVAL)
            else:
                raise Exception(request["jobStatus"])
        else:
            return bool(j["results"] or j["failedIds"])


def get_batch(batch_response, file_format, compressed):
    batch_url = get_next_link(batch_response.headers)
    while batch_url:
        batch_response = session.get(batch_url)
        batch_response.raise_for_status()
        yield decode_results(batch_response, file_format, compressed)
        batch_url = get_next_link(batch_response.headers)


def combine_batches(all_results, batch_results, file_format):
    if file_format == "json":
        for key in ("results", "failedIds"):
            if key in batch_results and batch_results[key]:
                all_results[key] += batch_results[key]
    elif file_format == "tsv":
        return all_results + batch_results[1:]
    else:
        return all_results + batch_results
    return all_results


def get_id_mapping_results_link(job_id):
    url = f"{API_URL}/idmapping/details/{job_id}"
    request = session.get(url)
    check_response(request)
    return request.json()["redirectURL"]


def decode_results(response, file_format, compressed):
    if compressed:
        decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
        if file_format == "json":
            j = json.loads(decompressed.decode("utf-8"))
            return j
        elif file_format == "tsv":
            return [line for line in decompressed.decode("utf-8").split("\n") if line]
        elif file_format == "xlsx":
            return [decompressed]
        elif file_format == "xml":
            return [decompressed.decode("utf-8")]
        else:
            return decompressed.decode("utf-8")
    elif file_format == "json":
        return response.json()
    elif file_format == "tsv":
        return [line for line in response.text.split("\n") if line]
    elif file_format == "xlsx":
        return [response.content]
    elif file_format == "xml":
        return [response.text]
    return response.text


def get_xml_namespace(element):
    matched = re.match(r"\{(.*)\}", element.tag)
    return matched.groups()[0] if matched else ""


# TO DO: remove this later if indeed not used anywhere
def merge_xml_results(xml_results):
    merged_root = ET.fromstring(xml_results[0])
    for result in xml_results[1:]:
        root = ET.fromstring(result)
        for child in root.findall("{http://uniprot.org/uniprot}entry"):
            merged_root.insert(-1, child)
    ET.register_namespace("", get_xml_namespace(merged_root[0]))
    return ET.tostring(merged_root, encoding="utf-8", xml_declaration=True)


def write_xml_results(xml_results: List[str], folder: str):
    # save individual xml files from the result of the REST query
    for xml_str in xml_results:
        root = ET.fromstring(xml_str)

        # repeat xml header (two lines) and closing tag (last line) on each entry
        l1 = xml_str.find('\n') + 1
        l2 = xml_str.find('\n', l1) + 1
        header = xml_str[:l2]
        header_bin = header.encode('ascii')

        root_tag = root.tag.split("}")[-1]
        last_line = f"</{root_tag}>"
        #last = xml_str.rfind('\n') + 1
        #last_line = xml_str[last:]
        last_line_bin = last_line.encode('ascii')

        # repeat xml namespace on each entry
        ET.register_namespace("", get_xml_namespace(root[0]))
        
        for entry in root.findall("{http://uniprot.org/uniprot}entry"):
            seqname = entry.findtext('{http://uniprot.org/uniprot}name')
            filepath = folder + seqname + ".xml"
            tmp = ET.tostring(element = entry)
            with open(filepath, "wb") as f:
                f.write(header_bin + tmp + last_line_bin)


def print_progress_batches(batch_index, size, total):
    n_fetched = min((batch_index + 1) * size, total)
    print(f"Fetched: {n_fetched} / {total}")


def get_id_mapping_results_search(url, get_format: Optional[str] = None):
    if get_format is not None:
        url = url + "?format=" + get_format
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    if "size" in query:
        size = int(query["size"][0])
    else:
        size = 500
        query["size"] = size
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    parsed = parsed._replace(query=urlencode(query, doseq=True))
    url = parsed.geturl()
    request = session.get(url)
    check_response(request)
    results = decode_results(request, file_format, compressed)
    print(request.headers)
    total = int(request.headers["X-Total-Results"])
    print_progress_batches(0, size, total)
    for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
        results = combine_batches(results, batch, file_format)
        print_progress_batches(i, size, total)
    """
    # avoiding as it depends on python 3.10
    if file_format == "xml":
        return merge_xml_results(results)
    """
    return results


def get_id_mapping_results_stream(url):
    if "/stream/" not in url:
        url = url.replace("/results/", "/results/stream/")
    request = session.get(url)
    check_response(request)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    file_format = query["format"][0] if "format" in query else "json"
    compressed = (
        query["compressed"][0].lower() == "true" if "compressed" in query else False
    )
    return decode_results(request, file_format, compressed)
