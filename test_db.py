import traceback
from backend.services.data_service import list_datasets

try:
    print("Testing data_service.list_datasets()...")
    res = list_datasets()
    print("Success. Result:")
    print(res)
except Exception as e:
    print("Error:")
    traceback.print_exc()
