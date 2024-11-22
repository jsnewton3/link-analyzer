from flask import request, Blueprint, abort, jsonify, make_response
from werkzeug.exceptions import HTTPException
from controllers.monitoring_task_session_handler import Monitor_Task_Session_Handler
session_handler = Monitor_Task_Session_Handler()
import logging
logger = logging.getLogger(__name__)

monitor_blueprint = Blueprint("monitor_blueprint", __name__, url_prefix='/')
@monitor_blueprint.route("monitor/config", methods=['PUT'])
def configure_monitoring_task():
    logger.info("Configure monitoring task request received")
    data = request.json
    print(str(data))
    type = data.get('type')
    config = data.get('config')
    target = data.get('target')
    response_json = session_handler.add_monitor_task(type, config, target)
    if session_handler._err:
        msg = session_handler._err
        code = session_handler._err['code']
        abort(make_response(jsonify(msg), code))
    return response_json

@monitor_blueprint.route("monitor/start", methods=['GET'])
def start_monitoring_task():
    uuid = request.json['uuid']
    logger.info("Start monitoring request recieved from " + str(uuid))
    response_json = session_handler.start_monitoring(uuid)
    if session_handler._err:
        msg = session_handler._err
        code = session_handler._err['code']
        abort(make_response(jsonify(msg), code))
    return response_json

@monitor_blueprint.route("monitor/available_type", methods=['GET'])
def get_available_analyses():
    types = list(session_handler.analysis_type_dict.keys())
    num_keys = [i for i in range(len(types))]
    analysis_dict = { k:v for (k,v) in zip(num_keys, types)}
    if session_handler._err:
        msg = session_handler._err
        code = session_handler._err['code']
        abort(make_response(jsonify(msg), code))
    return analysis_dict