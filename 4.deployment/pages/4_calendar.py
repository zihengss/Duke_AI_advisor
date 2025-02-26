import streamlit as st
from streamlit_calendar import calendar

# Initialize session state for classes if not already set
if 'class_schedule' not in st.session_state:
    st.session_state.class_schedule = [
        {"title": "CS101 - Data Structures", "start": "2025-02-17T09:00:00", "end": "2025-02-17T10:30:00", "resourceId": "a"},
        {"title": "MATH201 - Linear Algebra", "start": "2025-02-17T11:00:00", "end": "2025-02-17T12:30:00", "resourceId": "a"},
        {"title": "PHYS101 - Mechanics", "start": "2025-02-17T13:00:00", "end": "2025-02-17T14:30:00", "resourceId": "a"},
        {"title": "ENG102 - Writing & Composition", "start": "2025-02-18T10:00:00", "end": "2025-02-18T11:30:00", "resourceId": "a"},
    ]

# Define calendar options
calendar_options = {
    "editable": False,
    "selectable": True,
    "headerToolbar": {
        "left": "today prev,next",
        "center": "title",
        "right": "resourceTimelineDay,resourceTimelineWeek,resourceTimelineMonth",
    },
    "slotMinTime": "06:00:00",
    "slotMaxTime": "22:00:00",
    "initialView": "resourceTimelineWeek",
    "resourceGroupField": "building",
    "resources": [
        {"id": "a", "building": "Building A", "title": "Room 101"},
    ],
}

# Load events from session state
calendar_events = st.session_state.class_schedule

custom_css = """
    .fc-event-past {
        opacity: 0.8;
    }
    .fc-event-time {
        font-style: italic;
    }
    .fc-event-title {
        font-weight: 700;
    }
    .fc-toolbar-title {
        font-size: 2rem;
    }
"""

# Display the calendar
calendar = calendar(
    events=calendar_events,
    options=calendar_options,
    key='calendar',
)
st.write(calendar)

# Optionally, allow users to add new classes
with st.form("add_class_form"):
    title = st.text_input("Class Name")
    start_time = st.text_input("Start Time (YYYY-MM-DDTHH:MM:00)")
    end_time = st.text_input("End Time (YYYY-MM-DDTHH:MM:00)")
    resource_id = "a"  # All classes in the same building
    submit_button = st.form_submit_button("Add Class")
    
    if submit_button and title and start_time and end_time:
        new_class = {
            "title": title,
            "start": start_time,
            "end": end_time,
            "resourceId": resource_id,
        }
        st.session_state.class_schedule.append(new_class)
        st.rerun()
