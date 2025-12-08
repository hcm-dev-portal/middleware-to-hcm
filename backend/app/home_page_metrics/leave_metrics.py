def _sql_leave_metrics(as_of: str) -> str:
    return f"""
WITH params AS (
  SELECT CAST('{as_of}' AS DATE) AS asOf
),
wk AS (
  SELECT
    asOf,
    ((DATEPART(weekday, asOf) + 5) % 7) AS w,
    DATEADD(day, -((DATEPART(weekday, asOf)+5)%7), asOf) AS weekStart,
    DATEADD(day,  6-((DATEPART(weekday, asOf)+5)%7), asOf) AS weekEnd
  FROM params
),
-- Source leave rows with normalized dates
leave_src AS (
  SELECT
    PERSONID,
    DEPARTMENTID,   -- keep for diagnostics only
    ATTENDANCETYPE,
    COALESCE(
      COALESCE(TRY_CONVERT(date, STARTDATE, 112), TRY_CONVERT(date, STARTDATE, 23), TRY_CONVERT(date, STARTDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112),  TRY_CONVERT(date, WORKDATE, 23),  TRY_CONVERT(date, WORKDATE))
    ) AS SDATE,
    COALESCE(
      COALESCE(TRY_CONVERT(date, ENDDATE, 112), TRY_CONVERT(date, ENDDATE, 23), TRY_CONVERT(date, ENDDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112), TRY_CONVERT(date, WORKDATE, 23), TRY_CONVERT(date, WORKDATE))
    ) AS EDATE,
    VALIDATED
  FROM [eHRAntung_DB].[dbo].[ATDLEAVEDATA]
),
-- Person → Branch (authoritative)
emp AS (
  SELECT
    p.PERSONID,
    p.TRUENAME                               AS person_name,
    CAST(p.BRANCHID AS NVARCHAR(100))        AS branch_id
  FROM [eHRAntung_DB].[dbo].[PSNACCOUNT] p
),
-- Branch dimension
org AS (
  SELECT
    CAST(o.UNITID AS NVARCHAR(100))          AS unit_id,
    COALESCE(o.UNITDISPLAYNAME, o.UNITNAME)  AS branch_name,
    o.UNITCODE                               AS branch_code,
    ISNULL(o.ISDELETE, 0)                    AS branch_is_deleted_flag
  FROM [eHRAntung_DB].[dbo].[ORGStdStruct] o
),
-- Who is on leave on asOf (resolved via person→branch)
on_leave_day AS (
  SELECT 
    l.PERSONID,
    e.person_name,
    l.DEPARTMENTID                 AS department_id_original,
    e.branch_id,
    o.branch_name,
    o.branch_code,
    o.branch_is_deleted_flag,
    l.ATTENDANCETYPE               AS type_code,
    l.EDATE
  FROM leave_src l
  JOIN emp e
    ON e.PERSONID = l.PERSONID
  LEFT JOIN org o
    ON e.branch_id = o.unit_id
  CROSS JOIN params p0
  WHERE l.SDATE <= p0.asOf AND l.EDATE >= p0.asOf
),
pending_reqs AS (
  SELECT COUNT(*) AS cnt
  FROM leave_src
  WHERE (VALIDATED IS NULL OR VALIDATED = 0)
),
-- Upcoming 7 days (also resolved to person→branch)
upcoming_next7 AS (
  SELECT
    l.PERSONID                AS person_id,
    e.person_name,
    e.branch_id,
    o.branch_name,
    o.branch_code,
    l.ATTENDANCETYPE          AS type_code,
    l.SDATE                   AS start_date,
    l.EDATE                   AS end_date
  FROM leave_src l
  JOIN emp e
    ON e.PERSONID = l.PERSONID
  LEFT JOIN org o
    ON e.branch_id = o.unit_id
  CROSS JOIN params p0
  WHERE l.SDATE BETWEEN DATEADD(day, 1, p0.asOf) AND DATEADD(day, 7, p0.asOf)
),
-- Department/branch summary (group by resolved branch)
dept_summary AS (
  SELECT 
    branch_id                AS department_id,         -- keep output key name for compatibility
    MAX(branch_code)         AS department_code,
    MAX(branch_name)         AS department_name,
    COUNT(*)                 AS [count]
  FROM on_leave_day
  GROUP BY branch_id
),
-- Diagnostics to validate mapping quality
dept_sanity AS (
  SELECT
    SUM(CASE WHEN branch_id IS NULL THEN 1 ELSE 0 END)      AS missing_branchid_in_psnaccount,
    SUM(CASE WHEN branch_id IS NOT NULL AND branch_name IS NULL THEN 1 ELSE 0 END) AS missing_org_for_branchid,
    COUNT(*) AS total_checked
  FROM on_leave_day
),
overtime_week AS (
  SELECT
    SUM(CAST(HOURS AS DECIMAL(10,2))) AS total_hours,
    COUNT(DISTINCT PERSONID)          AS people
  FROM [eHRAntung_DB].[dbo].[ATDHISOVERTIME]
  CROSS JOIN wk
  WHERE COALESCE(TRY_CONVERT(date, OVERTIMEDATE, 112), TRY_CONVERT(date, OVERTIMEDATE, 23), TRY_CONVERT(date, OVERTIMEDATE))
        BETWEEN wk.weekStart AND wk.weekEnd
),
low_balance AS (
  SELECT COUNT(*) AS low_cnt
  FROM (
    SELECT PERSONID, MIN(REMAINDAYS) AS rem
    FROM (
      SELECT PERSONID, REMAINDAYS FROM [eHRAntung_DB].[dbo].[ATDNONCALCULATEDVACATION]
      UNION ALL
      SELECT PERSONID, REMAINDAYS FROM [eHRAntung_DB].[dbo].[ATDHISNONCALCULATEDVACATION]
    ) X
    GROUP BY PERSONID
  ) Y
  WHERE TRY_CAST(rem AS DECIMAL(10,2)) < 5
)
SELECT
  1 AS success,
  (
    SELECT
      (SELECT COUNT(*) FROM on_leave_day)               AS employees_on_leave_today,
      (SELECT cnt FROM pending_reqs)                    AS pending_leave_requests,
      (SELECT low_cnt FROM low_balance)                 AS low_balance_count,
      (SELECT ISNULL(total_hours,0) FROM overtime_week) AS overtime_hours,
      (SELECT ISNULL(people,0) FROM overtime_week)      AS overtime_people,

      -- Include person + resolved branch in details
      (SELECT TOP (50)
         PERSONID                        AS person_id,
         person_name,
         type_code,
         CONVERT(date, EDATE)            AS end_date,
         department_id_original,         -- from ATDLEAVEDATA (for debugging)
         branch_id                       AS department_id,   -- keep key for compat
         COALESCE(branch_code,'')        AS department_code,
         COALESCE(branch_name,'')        AS department_name
       FROM on_leave_day
       ORDER BY PERSONID
       FOR JSON PATH)                                   AS on_leave_details,

      (SELECT
         person_id,
         person_name,
         CONVERT(date, start_date)       AS start_date,
         CONVERT(date, end_date)         AS end_date,
         type_code,
         branch_id                       AS department_id,
         COALESCE(branch_code,'')        AS department_code,
         COALESCE(branch_name,'')        AS department_name
       FROM upcoming_next7
       ORDER BY start_date, person_id
       FOR JSON PATH)                                   AS upcoming_leave,

      (SELECT 
         department_id,
         department_code,
         department_name,
         [count]
       FROM dept_summary
       ORDER BY [count] DESC
       FOR JSON PATH)                                   AS department_summary,

      -- Mapping diagnostics surfaced in payload
      (SELECT
         (SELECT missing_branchid_in_psnaccount FROM dept_sanity) AS missing_branchid_in_psnaccount,
         (SELECT missing_org_for_branchid FROM dept_sanity)       AS missing_org_for_branchid,
         (SELECT total_checked FROM dept_sanity)                  AS total_checked
       FOR JSON PATH, WITHOUT_ARRAY_WRAPPER)                      AS department_join_sanity
    FOR JSON PATH, WITHOUT_ARRAY_WRAPPER
  ) AS metrics;
"""




def _sql_leave_trend(as_of: str, days: int) -> str:
    days = max(1, min(int(days or 7), 31))
    return f"""
WITH params AS (
  SELECT CAST('{as_of}' AS DATE) AS asOf
),
s(d) AS (
  SELECT DATEADD(day, -({days}-1), asOf) FROM params
  UNION ALL
  SELECT DATEADD(day, 1, d)
  FROM s CROSS JOIN params
  WHERE d < (SELECT asOf FROM params)
),
leave_src AS (
  SELECT
    PERSONID,
    COALESCE(
      COALESCE(TRY_CONVERT(date, STARTDATE, 112), TRY_CONVERT(date, STARTDATE, 23), TRY_CONVERT(date, STARTDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112),  TRY_CONVERT(date, WORKDATE, 23),  TRY_CONVERT(date, WORKDATE))
    ) AS SDATE,
    COALESCE(
      COALESCE(TRY_CONVERT(date, ENDDATE, 112), TRY_CONVERT(date, ENDDATE, 23), TRY_CONVERT(date, ENDDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112), TRY_CONVERT(date, WORKDATE, 23), TRY_CONVERT(date, WORKDATE))
    ) AS EDATE,
    ATTENDANCETYPE
  FROM dbo.ATDLEAVEDATA
),
daily_leave_details AS (
  SELECT 
    s.d AS [date],
    l.PERSONID       AS person_id,
    l.ATTENDANCETYPE AS type_code
  FROM s
  LEFT JOIN leave_src l ON l.SDATE <= s.d AND l.EDATE >= s.d
  WHERE l.PERSONID IS NOT NULL
),
counts AS (
  SELECT 
    [date],
    COUNT(*) AS [count],
    (SELECT 
       person_id,
       type_code
     FROM daily_leave_details d2 
     WHERE d2.[date] = daily_leave_details.[date]
     FOR JSON PATH) AS people_on_leave
  FROM daily_leave_details
  GROUP BY [date]
),
all_dates AS (
  SELECT 
    s.d AS [date],
    COALESCE(c.[count], 0) AS [count],
    COALESCE(c.people_on_leave, '[]') AS people_on_leave
  FROM s
  LEFT JOIN counts c ON c.[date] = s.d
)
SELECT 1 AS success,
       (SELECT 
          CONVERT(date, [date]) AS [date], 
          [count],
          JSON_QUERY(people_on_leave) AS people_on_leave
        FROM all_dates 
        ORDER BY [date]
        FOR JSON PATH) AS trend
OPTION (MAXRECURSION 200);
"""
