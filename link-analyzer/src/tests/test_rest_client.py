from uuid import UUID

# @pytest.fixture
def test_monitor_arima_valid_config(client, test_jsons):
    test_json = test_jsons['arima_config.json']
    response = client.put("monitor/config", json=test_json)
    errors = []
    # Test if valid json
    try:
        resp_dict = response.json
        # Test for single key and value
        if len(resp_dict.keys()) != 1:
            errors.append('a')
        # Check for correct key
        if list(response.json.keys())[0] != "uuid":
            errors.append('b')
    except ValueError as e:
        errors.append(e)
    try:
        # Check for valid uuid
        uuid_to_test = resp_dict["uuid"]
        uuid_obj = UUID(uuid_to_test)
    except ValueError as e:
        errors.append("e")
    assert  len(errors)==0

def test_monitor_invalid_analysis_error(client, test_jsons):
    test_json = test_jsons['invalid_analysis.json']
    errors = []
    try:
        response = client.put("monitor/config", json=test_json)
        if response.status_code != 406:
            errors.append(1)
        if response.json['msg']!="Invalid analysis type":
            errors.append(2)
    except Exception as e:
        errors.append(3)
    assert len(errors)==0

def test_invalid_arima_config(client, test_jsons):
    test_json = test_jsons['arima_config_error.json']
    response = client.put("monitor/config", json=test_json)
    errors = []
    try:
        response = client.put("monitor/config", json=test_json)
        if response.status_code != 406:
            errors.append(1)
        if response.json['msg'] != "Error initializing Arima filter. Invalid config parameters":
            errors.append(2)
    except ValueError as e:
        errors.append(e)

    assert  len(errors)==0

def test_available_analyses(client):
    response = client.get("/monitor/available_type")
    errors = []
    try:
        resp_dict = response.json
        # Check for valid key-value pairs
        types = ['Arima', 'Exponenetial_Moving_Average', 'MovingAvarage']
        for k,v in resp_dict.items():
            if isinstance(int(k), int) and v in types:
                continue
            else:
                errors.append('e')

    except ValueError as e:
        errors.append(e)
    except Exception as e:
        errors.append(e)
    assert len(errors) == 0
