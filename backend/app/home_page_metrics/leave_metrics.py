


"""
DO NOT MODIFY - Chiuzu 08/27/2025
"""
# ---------- Dashboard Data SQL builders (metrics / trend) ----------

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
leave_src AS (
  SELECT
    PERSONID,
    DEPARTMENTID,
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
  FROM dbo.ATDLEAVEDATA
),
on_leave_day AS (
  SELECT l.PERSONID, l.DEPARTMENTID, l.ATTENDANCETYPE AS [type], l.EDATE
  FROM leave_src l
  CROSS JOIN params p
  WHERE l.SDATE <= p.asOf AND l.EDATE >= p.asOf
),
pending_reqs AS (
  SELECT COUNT(*) AS cnt
  FROM leave_src
  WHERE (VALIDATED IS NULL OR VALIDATED = 0)
),
upcoming_next7 AS (
  SELECT
    l.PERSONID AS person_id,
    l.ATTENDANCETYPE AS [type],
    l.SDATE AS start_date,
    l.EDATE AS end_date
  FROM leave_src l
  CROSS JOIN params p
  WHERE l.SDATE BETWEEN DATEADD(day, 1, p.asOf) AND DATEADD(day, 7, p.asOf)
),
dept_summary AS (
  SELECT DEPARTMENTID AS department_id, COUNT(*) AS [count]
  FROM on_leave_day
  GROUP BY DEPARTMENTID
),
overtime_week AS (
  SELECT
    SUM(CAST(HOURS AS DECIMAL(10,2))) AS total_hours,
    COUNT(DISTINCT PERSONID)          AS people
  FROM dbo.ATDHISOVERTIME
  CROSS JOIN wk
  WHERE COALESCE(TRY_CONVERT(date, OVERTIMEDATE, 112), TRY_CONVERT(date, OVERTIMEDATE, 23), TRY_CONVERT(date, OVERTIMEDATE))
        BETWEEN wk.weekStart AND wk.weekEnd
),
low_balance AS (
  SELECT COUNT(*) AS low_cnt
  FROM (
    SELECT PERSONID, MIN(REMAINDAYS) AS rem
    FROM (
      SELECT PERSONID, REMAINDAYS FROM dbo.ATDNONCALCULATEDVACATION
      UNION ALL
      SELECT PERSONID, REMAINDAYS FROM dbo.ATDHISNONCALCULATEDVACATION
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
      (SELECT TOP (50)
         PERSONID AS person_id, [type], CONVERT(date, EDATE) AS end_date
       FROM on_leave_day
       ORDER BY PERSONID
       FOR JSON PATH)                                   AS on_leave_details,
      (SELECT
         person_id, CONVERT(date, start_date) AS start_date,
         CONVERT(date, end_date)   AS end_date, [type]
       FROM upcoming_next7
       ORDER BY start_date, person_id
       FOR JSON PATH)                                   AS upcoming_leave,
      (SELECT department_id, [count]
       FROM dept_summary
       ORDER BY [count] DESC
       FOR JSON PATH)                                   AS department_summary
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
    COALESCE(
      COALESCE(TRY_CONVERT(date, STARTDATE, 112), TRY_CONVERT(date, STARTDATE, 23), TRY_CONVERT(date, STARTDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112),  TRY_CONVERT(date, WORKDATE, 23),  TRY_CONVERT(date, WORKDATE))
    ) AS SDATE,
    COALESCE(
      COALESCE(TRY_CONVERT(date, ENDDATE, 112), TRY_CONVERT(date, ENDDATE, 23), TRY_CONVERT(date, ENDDATE)),
      COALESCE(TRY_CONVERT(date, WORKDATE, 112), TRY_CONVERT(date, WORKDATE, 23), TRY_CONVERT(date, WORKDATE))
    ) AS EDATE
  FROM dbo.ATDLEAVEDATA
),
counts AS (
  SELECT s.d AS [date],
         (SELECT COUNT(*) FROM leave_src l WHERE l.SDATE <= s.d AND l.EDATE >= s.d) AS [count]
  FROM s
)
SELECT 1 AS success,
       (SELECT CONVERT(date, [date]) AS [date], [count]
        FROM counts ORDER BY [date]
        FOR JSON PATH) AS trend
OPTION (MAXRECURSION 200);
"""