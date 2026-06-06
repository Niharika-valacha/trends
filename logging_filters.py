# filters.py
import logging
from threading import local
from uuid import uuid4

class RequestLogFilter(logging.Filter):

    def filter(self, record):
        from middleware import get_current_request
        """Attach request context (log_id) to log records"""
        request = get_current_request()
        if request:
            record.log_id = request.log_id
        else:
            record.log_id = str(uuid4())
        return True
    
"""
** File Name:        logging_filters.py
** Author:           Pradhuman Tamboli
** Creation Date:    2025-02-06 
**
******************************************************************************
**                    COPYRIGHT                                             **
**                                                                          **
** (C) Copyright 2024                                                       **
** Cygnus Compliance Consulting, Inc.                                       **
**                                                                          **
** This software is furnished under a license for use only on a single      **
** computer system and may be copied only with the inclusion of the above   **
** copyright notice. This software or any other copies thereof, may not be  **
** provided or otherwise made available to any other person except for use  **
** on such system and to one who agrees to these license terms. Title and   **
** ownership of the software shall at all times remain in                   **
** Cygnus Compliance Consulting, Inc.                                       **
**                                                                          **
** The information in this software is subject to change without notice and **
** should not be construed as a commitment by                               **
** Cygnus Compliance Consulting, Inc.                                       **
******************************************************************************
                Maintenance History

-------------|----------|----------------------------------------------------
    Date     |  Person  |  Description of Modification
-------------|----------|----------------------------------------------------
2025-02-06    |  Pradhuman  |  Initial Creation 
-------------|----------|----------------------------------------------------
"""