/* RID:7f61584501034c23af9a873715930c16 STAGE:SQL_RESPONSE_FINAL */
SET NOCOUNT ON;
WITH L AS (
    SELECT 
        CAST(ld.PERSONID AS NVARCHAR(100)) AS person_id_norm,
        ld.DEPARTMENTID,
        SUM(COALESCE(ld.HOURS, 0)) AS total_leave_hours
    FROM eHRAntung_DB.dbo.atdleavedata ld
    WHERE ld.WORKDATE >= '2025-02-01' AND ld.WORKDATE < '2025-03-01' AND ld.VALIDATED = 1
    GROUP BY ld.PERSONID, ld.DEPARTMENTID
),
D AS (
    SELECT 
        CAST(org.UNITID AS NVARCHAR(100)) AS department_id_norm,
        COALESCE(org.UNITDISPLAYNAME, org.UNITNAME) AS department_name,
        SUM(L.total_leave_hours) AS total_hours
    FROM L
    LEFT JOIN [eHRAntung_DB].[dbo].[ORGStdStruct] org ON L.DEPARTMENTID = org.UNITID
    GROUP BY org.UNITID, org.UNITDISPLAYNAME, org.UNITNAME
)
SELECT 
    department_name AS 部門,
    total_hours AS 請假時數
FROM D
ORDER BY total_hours DESC
