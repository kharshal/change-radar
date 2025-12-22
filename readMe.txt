

change_radar/
├── app.py                    # Main entry point
├── config/
│   └── settings.py           # All configuration constants
├── components/
│   ├── header.py             # Reusable header
│   ├── feedback_modal.py     # Feedback dialog
│   └── kpi_card.py           # KPI card component
├── pages/
│   ├── build_knowledge.py    # Build Knowledge Layer (Steps 1–4)
│   ├── dashboard.py          # Main dashboard
│   └── deep_dive.py          # Deep dive page
├── services/
│   ├── data_service.py       # Data generation logic
│   └── kpi_service.py        # KPI extraction logic
└── utils/
    ├── session_state.py      # Session state manager
    └── styles.py             # CSS styles