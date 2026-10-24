# Copyright (c) Microsoft. All rights reserved.

import pytest


@pytest.mark.asyncio
async def test_s3_filesystem_paginates_writes_and_preserves_access_errors() -> None:
    s3fs = pytest.importorskip("s3fs")
    stub = pytest.importorskip("aiobotocore.stub")
    filesystem = s3fs.S3FileSystem(asynchronous=True, key="offline", secret="offline")
    client = await filesystem.set_session()
    try:
        with stub.AioStubber(client) as requests:
            params = {"Bucket": "training", "Prefix": "runs/", "Delimiter": "/"}
            requests.add_response(
                "list_objects_v2",
                {
                    "IsTruncated": True,
                    "NextContinuationToken": "next-page",
                    "Contents": [{"Key": "runs/one.json", "Size": 3}],
                },
                params,
            )
            requests.add_response(
                "list_objects_v2",
                {"IsTruncated": False, "Contents": [{"Key": "runs/two.json", "Size": 4}]},
                {**params, "ContinuationToken": "next-page"},
            )
            files = await filesystem._ls("training/runs", detail=True)
            assert [(item["name"], item["size"]) for item in files] == [
                ("training/runs/one.json", 3),
                ("training/runs/two.json", 4),
            ]
            requests.add_response(
                "put_object",
                {"ETag": '"saved"'},
                {"Bucket": "training", "Key": "runs/new.json", "Body": b"{}", "IfNoneMatch": "*"},
            )
            result = await filesystem._pipe_file("training/runs/new.json", b"{}", mode="create")
            assert result["ETag"] == '"saved"'
            requests.add_client_error(
                "put_object",
                service_error_code="AccessDenied",
                service_message="Denied",
                http_status_code=403,
                expected_params={"Bucket": "training", "Key": "blocked.json", "Body": b"{}"},
            )
            with pytest.raises(PermissionError):
                await filesystem._pipe_file("training/blocked.json", b"{}")
            requests.assert_no_pending_responses()
    finally:
        await client.close()
