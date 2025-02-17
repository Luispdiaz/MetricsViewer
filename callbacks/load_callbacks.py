from dash import Input, Output, State, html, no_update
from data.data_loader import parse_contents 

def register_load_callbacks(app):
    @app.callback(
        Output('output-data-upload', 'children'),        
        Output('stored-file-info', 'data'),                
        Output('stored-loss-data-total', 'data'),          
        Output('stored-loss-data-output', 'data'),         
        Output('stored-pet-train', 'data'),               
        Output('stored-pet-valid', 'data'),              
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        prevent_initial_call=True
    )
    def update_output(contents, filenames):
        if contents and filenames:
            (files_list,
             model_info,
             model_data_loss_total,
             model_data_loss_output,
             model_pet_train,
             model_pet_valid) = parse_contents(contents, filenames)
            return files_list, model_info, model_data_loss_total, model_data_loss_output, model_pet_train, model_pet_valid
        
        return html.Div(['No files have been uploaded.']), no_update, no_update, no_update, no_update, no_update

    @app.callback(
        Output('check-files-result', 'children'), 
        Output('url', 'pathname'),                
        Input('check-files-button', 'n_clicks'),      
        State('upload-data', 'contents'),           
        prevent_initial_call=True
    )
    def check_files(n_clicks, contents):
        if n_clicks is None:
            return no_update, no_update
        if contents:
            return html.Div(), '/dashboard'  
        else:
            return html.Div('No files have been uploaded.',
                            style={'color': 'red', 'margin-top': '20px'}), no_update
