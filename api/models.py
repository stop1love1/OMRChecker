"""
API models for documentation and serialization
"""
from flask_restx import fields

def setup_models(api):
    """Initialize API models for swagger documentation"""
    
    template_model = api.model('Template', {
        'description': fields.String(description='Template description'),
        'author': fields.String(description='Template author'),
    })

    omr_result_model = api.model('OMRResult', {
        'file_name': fields.String(description='Image file name'),
        'result_id': fields.String(description='Unique result identifier'),
        'message': fields.String(description='Processing status message'),
    })

    public_image_model = api.model('PublicImage', {
        'url': fields.String(description='URL to access image from outside')
    })

    result_model = api.model('Result', {
        'file_id': fields.String(description='Original image filename'),
        'score': fields.Float(description='Score'),
        'studentId': fields.String(description='Student ID'),
        'code': fields.String(description='Code or identifier'),
        'input_image_url': fields.String(description='URL of input image'),
        'output_image_url': fields.String(description='URL of processed image'),
        'answers': fields.List(fields.Nested(api.model('Answer', {
            'key': fields.String(description='Question code'),
            'value': fields.String(description='Answer value')
        }))),
    })
    
    return {
        'template_model': template_model,
        'omr_result_model': omr_result_model,
        'public_image_model': public_image_model,
        'result_model': result_model
    } 