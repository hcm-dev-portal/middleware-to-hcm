# app/leave/service.py
from typing import Dict, Any, Optional, List
import httpx
import json
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

# ===== REQUEST MODELS =====

class LeaveDetailRequest(BaseModel):
    employeeid: str
    leavetypeid: str
    begindate: str  # YYYY-MM-DD format
    begintime: str  # HH:MM format
    enddate: str    # YYYY-MM-DD format
    endtime: str    # HH:MM format
    IsRepeatLeave: str = "0"

class LeaveFormRequest(BaseModel):
    formno: str
    detail: List[LeaveDetailRequest]
    reason: Optional[str] = None

class LeaveRequest(BaseModel):
    user_id: str = Field(..., description="Login name for authentication")
    employee_id: str = Field(..., description="Employee ID from HCM system")
    leave_type: str = Field(..., description="Type of leave (annual, sick, personal, emergency)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    start_time: str = Field(default="09:00", description="Start time in HH:MM format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    end_time: str = Field(default="18:00", description="End time in HH:MM format")
    reason: Optional[str] = Field(default="", description="Reason for leave")
    duration_type: str = Field(default="full-day", description="full-day, half-day-am, half-day-pm, multiple-days")

class LeaveBalanceRequest(BaseModel):
    user_id: str
    employee_id: str

# ===== RESPONSE MODELS =====

class LeaveResponse(BaseModel):
    success: bool
    message: str
    request_id: Optional[str] = None
    form_number: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class LeaveBalanceResponse(BaseModel):
    success: bool
    balances: Dict[str, float]
    employee_id: str

# ===== HCM API SERVICE =====

class HCMLeaveService:
    def __init__(self):
        self.base_url = "https://qgaia.royal.club.tw/eHR/eHRExternalService/service.ashx"
        self.access_token = "1DA1FAD6-6183-4174-8321-E8B853EA8D2D"
        self.business_unit = "0"
        self.region = "zh-CN"
        
        # Leave type mapping
        self.leave_type_mapping = {
            "annual": "19",      # Annual leave
            "sick": "20",        # Sick leave  
            "personal": "21",    # Personal leave
            "emergency": "22"    # Emergency leave
        }
        
        # Form number mapping for different services
        self.form_numbers = {
            "create_leave": "1001",
            "create_form_leave": "10003",
            "apply_form_leave": "10003"
        }

    def _build_logon_info(self, service_code: str, login_name: str = "sa") -> str:
        """Build the LogonInfo string for HCM API"""
        expire_date = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        return f"LoginName={login_name}&BusinessUnit={self.business_unit}&LogonRegion={self.region}&ExpiredDate={expire_date}&ServiceCode={service_code}"

    def _map_duration_to_times(self, duration_type: str, start_date: str) -> tuple:
        """Map duration type to appropriate start/end times"""
        time_mappings = {
            "full-day": ("09:00", "18:00"),
            "half-day-am": ("09:00", "12:00"),
            "half-day-pm": ("13:00", "18:00"),
            "multiple-days": ("09:00", "18:00")  # Default for multi-day
        }
        return time_mappings.get(duration_type, ("09:00", "18:00"))

    async def create_leave_request(self, request: LeaveRequest) -> LeaveResponse:
        """Create a leave request using the HCM API"""
        try:
            # Map leave type to HCM system ID
            leave_type_id = self.leave_type_mapping.get(request.leave_type, "19")
            
            # Adjust times based on duration type
            if request.duration_type in ["full-day", "half-day-am", "half-day-pm"]:
                start_time, end_time = self._map_duration_to_times(request.duration_type, request.start_date)
            else:
                start_time, end_time = request.start_time, request.end_time

            # Step 1: Create form leave first
            form_response = await self._create_form_leave(
                employee_id=request.employee_id,
                leave_type_id=leave_type_id,
                start_date=request.start_date,
                start_time=start_time,
                end_date=request.end_date,
                end_time=end_time,
                login_name=request.user_id
            )
            
            if not form_response["success"]:
                return LeaveResponse(
                    success=False,
                    message=f"Failed to create leave form: {form_response.get('message', 'Unknown error')}"
                )

            # Step 2: Apply the form leave
            apply_response = await self._apply_form_leave(
                employee_id=request.employee_id,
                leave_type_id=leave_type_id,
                start_date=request.start_date,
                start_time=start_time,
                end_date=request.end_date,
                end_time=end_time,
                reason=request.reason or "Leave request submitted via AI assistant",
                login_name=request.user_id
            )

            if apply_response["success"]:
                return LeaveResponse(
                    success=True,
                    message="Leave request submitted successfully",
                    request_id=f"LR-{datetime.now().year}-{datetime.now().strftime('%m%d%H%M')}",
                    form_number=self.form_numbers["apply_form_leave"],
                    data={
                        "employee_id": request.employee_id,
                        "leave_type": request.leave_type,
                        "start_date": request.start_date,
                        "end_date": request.end_date,
                        "duration": request.duration_type,
                        "reason": request.reason
                    }
                )
            else:
                return LeaveResponse(
                    success=False,
                    message=f"Failed to apply leave form: {apply_response.get('message', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"Error creating leave request: {str(e)}")
            return LeaveResponse(
                success=False,
                message=f"System error: {str(e)}"
            )

    async def _create_form_leave(self, employee_id: str, leave_type_id: str, 
                                start_date: str, start_time: str,
                                end_date: str, end_time: str, login_name: str = "sa") -> Dict[str, Any]:
        """Create a leave form using createformleave service"""
        
        payload = {
            "AccessToken": self.access_token,
            "LogonInfo": self._build_logon_info("createformleave", login_name),
            "Data": {
                "formno": self.form_numbers["create_form_leave"],
                "detail": [
                    {
                        "employeeid": employee_id,
                        "leavetypeid": leave_type_id,
                        "begindate": start_date,
                        "begintime": start_time,
                        "enddate": end_date,
                        "endtime": end_time
                    }
                ]
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Create form leave response: {result}")
                
                # Check if the response indicates success
                # You may need to adjust this based on the actual API response format
                return {
                    "success": True,
                    "message": "Leave form created successfully",
                    "data": result
                }
                
        except httpx.RequestError as e:
            logger.error(f"Request error in create_form_leave: {str(e)}")
            return {"success": False, "message": f"Request error: {str(e)}"}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in create_form_leave: {e.response.status_code}")
            return {"success": False, "message": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Unexpected error in create_form_leave: {str(e)}")
            return {"success": False, "message": f"Unexpected error: {str(e)}"}

    async def _apply_form_leave(self, employee_id: str, leave_type_id: str,
                               start_date: str, start_time: str, 
                               end_date: str, end_time: str, reason: str,
                               login_name: str = "sa") -> Dict[str, Any]:
        """Apply/submit a leave form using applyformleave service"""
        
        payload = {
            "AccessToken": self.access_token,
            "LogonInfo": self._build_logon_info("applyformleave", login_name),
            "Data": {
                "formno": self.form_numbers["apply_form_leave"],
                "reason": reason,
                "detail": [
                    {
                        "employeeid": employee_id,
                        "leavetypeid": leave_type_id,
                        "begindate": start_date,
                        "begintime": start_time,
                        "enddate": end_date,
                        "endtime": end_time
                    }
                ]
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Apply form leave response: {result}")
                
                return {
                    "success": True,
                    "message": "Leave application submitted successfully",
                    "data": result
                }
                
        except httpx.RequestError as e:
            logger.error(f"Request error in apply_form_leave: {str(e)}")
            return {"success": False, "message": f"Request error: {str(e)}"}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in apply_form_leave: {e.response.status_code}")
            return {"success": False, "message": f"HTTP error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"Unexpected error in apply_form_leave: {str(e)}")
            return {"success": False, "message": f"Unexpected error: {str(e)}"}

    async def get_leave_balance(self, request: LeaveBalanceRequest) -> LeaveBalanceResponse:
        """Get leave balance for an employee (mock implementation for now)"""
        # Note: The get-leave-hours endpoint is marked as -N (cannot work)
        # So we'll return mock data for now, but you can implement actual logic
        # when you have a working endpoint
        
        try:
            # Mock balances - replace with actual API call when available
            mock_balances = {
                "annual": 15.5,
                "sick": 8.0,
                "personal": 3.0,
                "emergency": 0.0  # Usually unlimited
            }
            
            return LeaveBalanceResponse(
                success=True,
                balances=mock_balances,
                employee_id=request.employee_id
            )
            
        except Exception as e:
            logger.error(f"Error getting leave balance: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get leave balance: {str(e)}")

# ===== SERVICE INSTANCE =====

hcm_leave_service = HCMLeaveService()

# ===== MAIN SERVICE FUNCTIONS =====

async def submit_leave_request(request: LeaveRequest) -> LeaveResponse:
    """Main function to submit a leave request"""
    return await hcm_leave_service.create_leave_request(request)

async def get_employee_leave_balance(request: LeaveBalanceRequest) -> LeaveBalanceResponse:
    """Get employee leave balance"""
    return await hcm_leave_service.get_leave_balance(request)

async def validate_leave_request(request: LeaveRequest) -> Dict[str, Any]:
    """Validate leave request before submission"""
    errors = []
    warnings = []
    
    # Basic validation
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        if start_date < datetime.now().date():
            if request.leave_type != "sick":
                warnings.append("Leave start date is in the past")
        
        if end_date < start_date:
            errors.append("End date cannot be before start date")
            
        if request.leave_type not in ["annual", "sick", "personal", "emergency"]:
            errors.append("Invalid leave type")
            
    except ValueError:
        errors.append("Invalid date format. Use YYYY-MM-DD")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }