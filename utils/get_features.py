def get_features(case_study):
    """
    Returns the features for the given case study.

    Parameters:
    case_study (str): The case study identifier.

    Returns:
    tuple: A tuple containing the case ID name, activity column name, resource column name, 
           continuous features, start date name, and end date name.
    """
    if case_study == "bac":
        continuous_features = ['time_from_start', 'time_from_previous_event(start)', 'time_from_midnight', 'activity_duration', 
                               '# ACTIVITY=Service closure Request with network responsibility', 
                               '# ACTIVITY=Service closure Request with BO responsibility', 
                               '# ACTIVITY=Pending Request for Reservation Closure', 
                               '# ACTIVITY=Pending Liquidation Request', 
                               '# ACTIVITY=Request completed with account closure', 
                               '# ACTIVITY=Request created', '# ACTIVITY=Authorization Requested', 
                               '# ACTIVITY=Evaluating Request (NO registered letter)', 
                               '# ACTIVITY=Network Adjustment Requested', '# ACTIVITY=Pending Request for acquittance of heirs', 
                               '# ACTIVITY=Request deleted', '# ACTIVITY=Back-Office Adjustment Requested', 
                               '# ACTIVITY=Evaluating Request (WITH registered letter)', 
                               '# ACTIVITY=Request completed with customer recovery', 
                               '# ACTIVITY=Pending Request for Network Information']
        case_id_name = 'REQUEST_ID'
        activity_column_name = 'ACTIVITY'
        resource_column_name = 'CE_UO'
        start_date_name = "start:timestamp"
        end_date_name = "end:timestamp"

    elif case_study == "bpi17_before":
        continuous_features = ['time_from_start', 'time_from_previous_event(start)', 'time_from_midnight', 'event_duration', 
                               '# ACTIVITY=O_Cancelled', '# ACTIVITY=O_Created', '# ACTIVITY=O_Sent (mail and online)', 
                               '# ACTIVITY=O_Sent (online only)', '# ACTIVITY=A_Submitted', '# ACTIVITY=A_Concept', 
                               '# ACTIVITY=A_Incomplete', '# ACTIVITY=O_Refused', '# ACTIVITY=A_Pending', '# ACTIVITY=A_Cancelled', 
                               '# ACTIVITY=A_Denied', '# ACTIVITY=A_Accepted', '# ACTIVITY=O_Returned', '# ACTIVITY=A_Validating', 
                               '# ACTIVITY=A_Create Application', '# ACTIVITY=O_Accepted', '# ACTIVITY=O_Create Offer', 
                               '# ACTIVITY=A_Complete']
        case_id_name = 'case:concept:name'
        activity_column_name = 'concept:name'
        resource_column_name = 'org:resource'
        start_date_name = "start:timestamp"
        end_date_name = "end:timestamp"

    elif case_study == "bpi17_after":
        continuous_features = ['time_from_start', 'time_from_previous_event(start)', 'time_from_midnight', 'event_duration', 
                               '# ACTIVITY=A_Concept', '# ACTIVITY=O_Sent (online only)', '# ACTIVITY=O_Create Offer', 
                               '# ACTIVITY=A_Complete', '# ACTIVITY=A_Accepted', '# ACTIVITY=A_Incomplete', '# ACTIVITY=A_Submitted', 
                               '# ACTIVITY=A_Cancelled', '# ACTIVITY=O_Refused', '# ACTIVITY=O_Accepted', '# ACTIVITY=O_Cancelled', 
                               '# ACTIVITY=O_Created', '# ACTIVITY=A_Validating', '# ACTIVITY=A_Pending', '# ACTIVITY=O_Returned', 
                               '# ACTIVITY=O_Sent (mail and online)', '# ACTIVITY=A_Create Application', '# ACTIVITY=A_Denied', 
                               'RequestedAmount']
        case_id_name = 'case:concept:name'
        activity_column_name = 'concept:name'
        resource_column_name = 'org:resource'
        start_date_name = "start:timestamp"
        end_date_name = "end:timestamp"

    elif case_study == "bpi13":
        continuous_features = ['time_from_start', 'time_from_previous_event(start)', 'time_from_midnight', 
                               '# ACTIVITY=Wait - User', '# ACTIVITY=In Call', '# ACTIVITY=Wait - Implementation', 
                               '# ACTIVITY=Awaiting Assignment', '# ACTIVITY=Resolved', '# ACTIVITY=In Progress', 
                               '# ACTIVITY=Assigned', '# ACTIVITY=Cancelled', '# ACTIVITY=Wait - Customer', 
                               '# ACTIVITY=Wait - Vendor', '# ACTIVITY=Wait', '# ACTIVITY=Closed', '# ACTIVITY=Unmatched']
        case_id_name = "SR_Number"
        activity_column_name = 'ACTIVITY'
        resource_column_name = "Involved_ST"
        start_date_name = "start:timestamp"
        end_date_name = "end:timestamp"

    elif case_study == "consulta":
        continuous_features = ['time_from_start', 'time_from_previous_event(start)', 'time_from_midnight', 'event_duration', 
                               '# ACTIVITY=Cancelar Solicitud', '# ACTIVITY=Transferir creditos homologables', 
                               '# ACTIVITY=Validar solicitud', '# ACTIVITY=Visto Bueno Cierre Proceso', 
                               '# ACTIVITY=Transferir Creditos', '# ACTIVITY=Notificacion estudiante cancelacion soli', 
                               '# ACTIVITY=Traer informacion estudiante - banner', '# ACTIVITY=Homologacion por grupo de cursos', 
                               '# ACTIVITY=Avanzar recepcion documentos', '# ACTIVITY=Radicar Solicitud Homologacion', 
                               '# ACTIVITY=Validacion final', '# ACTIVITY=Validar solicitud / pre-homologacion', 
                               '# ACTIVITY=Evaluacion curso', '# ACTIVITY=Revisar curso', '# ACTIVITY=Cancelar curso', 
                               '# ACTIVITY=Recepcion de documentos']
        case_id_name = 'case:concept:name'
        activity_column_name = 'concept:name'
        resource_column_name = 'org:resource'
        start_date_name = "start:timestamp"
        end_date_name = "end:timestamp"

    return case_id_name, activity_column_name, resource_column_name, continuous_features, start_date_name, end_date_name 