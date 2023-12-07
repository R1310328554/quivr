from secrets import token_hex
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends
from logger import get_logger
from middlewares.auth import AuthBearer, get_current_user
from modules.api_key.dto.outputs import ApiKeyInfo
from modules.api_key.entity.api_key import ApiKey
from modules.api_key.repository.api_keys import ApiKeys
from modules.user.entity.user_identity import UserIdentity

logger = get_logger(__name__)


api_key_router = APIRouter()

api_keys_repository = ApiKeys()


@api_key_router.post(
    "/api-key",
    response_model=ApiKey,
    dependencies=[Depends(AuthBearer())],
    tags=["API Key"],
)
async def create_api_key(current_user: UserIdentity = Depends(get_current_user)):
    """
    Create new API key for the current user.

    - `current_user`: The current authenticated user.
    - Returns the newly created API key.

    This endpoint generates a new API key for the current user. The API key is stored in the database and associated with
    the user. It returns the newly created API key.
    """

    new_key_id = uuid4()
    new_api_key = token_hex(16)

    try:
        # Attempt to insert new API key into database
        response = api_keys_repository.create_api_key(
            new_key_id, new_api_key, current_user.id, "api_key", 30, False
        )
    except Exception as e:
        logger.error(f"Error creating new API key: {e}")
        return {"api_key": "Error creating new API key."}
    logger.info(f"Created new API key for user {current_user.email}.")

    return response  # type: ignore


@api_key_router.delete(
    "/api-key/{key_id}", dependencies=[Depends(AuthBearer())], tags=["API Key"]
)
async def delete_api_key(
    key_id: str, current_user: UserIdentity = Depends(get_current_user)
):
    """
    Delete (deactivate) an API key for the current user.

    - `key_id`: The ID of the API key to delete.

    This endpoint deactivates and deletes the specified API key associated with the current user. The API key is marked
    as inactive in the database.

    """
    api_keys_repository.delete_api_key(key_id, current_user.id)

    return {"message": "API key deleted."}


@api_key_router.get(
    "/api-keys",
    response_model=List[ApiKeyInfo],
    dependencies=[Depends(AuthBearer())],
    tags=["API Key"],
)
async def get_api_keys(current_user: UserIdentity = Depends(get_current_user)):
    """
    Get all active API keys for the current user.

    - `current_user`: The current authenticated user.
    - Returns a list of active API keys with their IDs and creation times.

    This endpoint retrieves all the active API keys associated with the current user. It returns a list of API key objects
    containing the key ID and creation time for each API key.
    
    
    {
  "brains": [
    {
      "id": "22eca0ed-f0a9-4c14-9473-815808b207d9",
      "name": "Default brain",
      "rights": "Owner",
      "status": "private",
      "brain_type": "doc"
    }
  ]
}


{
  "knowledges": [
    {
      "id": "73deb1bb-ca42-4534-ba10-52d59e11c9d0",
      "brain_id": "22eca0ed-f0a9-4c14-9473-815808b207d9",
      "file_name": "submission.csv",
      "url": null,
      "extension": ".csv"
    },
    {
      "id": "742ddf83-f0ad-4226-a29f-e1b84db7677e",
      "brain_id": "22eca0ed-f0a9-4c14-9473-815808b207d9",
      "file_name": "067042.xls",
      "url": null,
      "extension": ".xls"
    },
    {
      "id": "77a432e6-8488-46d6-9899-12fc474c23b6",
      "brain_id": "22eca0ed-f0a9-4c14-9473-815808b207d9",
      "file_name": "032030.xls",
      "url": null,
      "extension": ".xls"
    }
  ]
}


{
  "chats": [
    {
      "chat_id": "3bda81a4-b806-4b68-abcf-eb183fa410af",
      "user_id": "848805ca-1633-417b-a575-0f8b9584986b",
      "creation_time": "2023-11-25T08:48:37.309254",
      "chat_name": "whats the SalePrice"
    },
    {
      "chat_id": "affcd006-512e-4fa0-8513-1a789eed291a",
      "user_id": "848805ca-1633-417b-a575-0f8b9584986b",
      "creation_time": "2023-11-26T12:03:19.358179",
      "chat_name": "who is bill"
    },
    {
      "chat_id": "1bbe12e5-3550-4e23-ad04-bb416a87f276",
      "user_id": "848805ca-1633-417b-a575-0f8b9584986b",
      "creation_time": "2023-11-26T15:17:26.480239",
      "chat_name": "New Chat"
    },
    {
      "chat_id": "419e3b3c-1b30-4e5b-b8e2-ac58e17806c2",
      "user_id": "848805ca-1633-417b-a575-0f8b9584986b",
      "creation_time": "2023-11-27T14:08:39.762102",
      "chat_name": "who is flanklin"
    },
    
    {
      "chat_id": "1936351d-45b0-4488-a037-9cbad617e06e",
      "user_id": "848805ca-1633-417b-a575-0f8b9584986b",
      "creation_time": "2023-11-30T12:03:54.034668",
      "chat_name": "1+2+3+...+99+1000"
    }
  ]
}


POST http://localhost:5050/chat/1936351d-45b0-4488-a037-9cbad617e06e/question/stream?brain_id=22eca0ed-f0a9-4c14-9473-815808b207d9

{"question":"1/2/3.../99/100","brain_id":"22eca0ed-f0a9-4c14-9473-815808b207d9"}


... why 
INFO:     127.0.0.1:62052 - "GET /chat/%7Bself.chat_id%7D/question HTTP/1.1" 405 Method Not Allowed

INFO:     127.0.0.1:50034 - "GET /api-keys HTTP/1.1" 403 Forbidden
INFO:     127.0.0.1:50613 - "POST /api-key HTTP/1.1" 403 Forbidden


create 
    {
  "api_key": "e2da85ad0ac654c74fbfb6e83de8dcd8",
  "key_id": "e83742b4-663a-437c-b3cc-50c9e8f91934"
}



  File "D:\d\git\gpt\Quivr\backend\routes\api_key_routes.py", line 107, in get_api_keys
    return response.data
           ^^^^^^^^^^^^^
AttributeError: 'list' object has no attribute 'data'






litellm.exceptions.ContextWindowExceededError: This model's maximum context length is 4097 tokens. However, you requested 111766 tokens (656 in the messages, 111110 in the completion). Please reduce the length of the messages or completion.





3fa85f64-5717-4562-b3fc-2c963f66afa6

    """
    response = api_keys_repository.get_user_api_keys(current_user.id)
    return response.data
