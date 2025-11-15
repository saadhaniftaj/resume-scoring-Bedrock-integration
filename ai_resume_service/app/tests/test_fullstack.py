# app/tests/test_fullstack.py
from app.processors.ai_matcher import analyze_resume_with_job

if __name__ == "__main__":
    job_description = """
We are seeking a Junior Full-Stack Developer (Co-Op) to help develop and maintain web applications, APIs, databases, and AI-driven features.

Qualifications:
- Proficiency with Node.js
- Proficiency with React
- Proficiency with MySQL
- Proficiency with HTML/CSS
- Proficiency with GitHub

Preferred:
- Bootstrap
- CI/CD (GitHub Actions/GitLab CI)
- Google Cloud Platform
- Mocha/Chai (TDD)
- iOS/Android app development
- Object-oriented programming (OOP)
- Agile/Scrum
- Strong communication and time-management
- Legally eligible to work in Canada
    """

    resumes = [
        ("Provided Resume", """
Technical Skills
Languages: C#, C++, Python, and Java
Web development: HTML, CSS, NodeJS, Ruby-on-rails, React, Angular, NgRx, JavaScript, jQuery, and TypeScript
Databases: PostgreSQL, MySQL and Microsoft SQL
Operating Systems: Windows, Mac OS X, and Linux
Mobile: iOS & Android
Version Control Systems: Jira, GitHub and Gitlab
Platforms: Google Cloud Platform and Kubernetes Engine

Work Experience
Full Stack Software Developer                                2021 to Present
District of North Vancouver – North Vancouver, BC
Worked on the internal Report a Problem web application used by employees to log problems
Helped develop multiple features such as saved searches, improved searches, and linked cases
Received the Municipal World Innovation Award alongside my team for the work done on internal RAP
Developed the public front-end Report a Problem web application for citizens to report problems
Developed the District’s comprehensive employee timesheet management system
Developed the council vote management software for the district’s staff use
Built user timesheet automation software to automate timesheet submissions and approvals for staff
Reconfiguring the automated permit integration software for automatic permit application generation

IT Consultant                                             2015 to 2021
GRD Construction Ltd.
Providing onsite support for hardware and software.
Building and configuring back-end storage servers.
Providing consultations on company's hardware and software requirements.

Analytics Cloud Web Developer                              09/2020 to 04/2021
SAP SE Vancouver – Vancouver, BC
Fixed bugs as a JavaScript web developer on the SAP Cloud Platform for the Story container
Worked on new performance improvement features for SAC filters
Team quality lead representative in the absence of our fulltime quality lead
Performed cross feature testing to ensure compatibility with features from different teams
Created test plans and implementation tests for various features using the Jasmine framework

Analytics Cloud Web Developer                              09/2018 to 04/2019
SAP SE Vancouver – Vancouver, BC
Contributed as a JavaScript web developer on SAP’s Cloud Platform for Boardroom.
Helped develop features such as infinite scroll for filters and advanced tooltips on data point popups
Fixed numerous internal bugs as well as various urgent customer bugs
Implemented an algorithm to fix display scaling on Boardroom regardless of the display resolution
Created the test plan and implementation tests for infinite scrolling feature on filters

Education
Software Systems                                            09/2016 to 08/2021
Simon Fraser University - Burnaby, BC
Bachelor of Science in Software Systems

Additional Skills
Effective organization and time management skills
Able to handle multiple tasks and responsibilities concurrently
Excellent problem-solving skills
Quick learner with positive attitude towards learning
Ability to work independently with minimum supervision
Team player and work well with others
Excellent communication skills using multiple languages. Fluent in English and Punjabi

Volunteer Work
Miracles of Technology
Worked with a team of SAP colleagues to demonstrate programming to grade 5-6 students to get them excited about the technology field.
We taught them how to code using a Raspberry Pi and manipulate a light strip using external switches. The article about this volunteer activity was posted in the SAP news portal.

SAP Facilities Manager Hackathon Project
The project records live data from a raspberry pi and motion sensor to determine which facilities are currently in use. It would upload the data to a database hosted on GCP which allowed us to query it on both our front-end web UI and our Boardroom presentation

Cradle VSA Web Application
Developed a web application to be used alongside with the Cradle VSA application on Android
Enables health care workers to identify and monitor complications surrounding pregnant women in Uganda
The project was done using Angular, CSS and JavaScript for the frontend, Java and the Hibernate Framework for the backend as well as PostgreSQL for the database
This was a team of 8 members where I contributed as a full-stack developer

SFU Venture
Developed a web application that allows students to buy and sell textbooks as well as track events
Utilizes SFU’s public APIs to query the faculties, sections, as well as required textbooks for each offered class
Uses login authentication comprised of JSON-Web-Tokens and email verification using Nodemailer
The project was built using Angular, SCSS, JavaScript and jQuery for the frontend, NodeJS for the backend and PostgreSQL for the database
This was a team of 5 members where I contributed as the full stack developer. I was also responsible for compiling, building and deploying the entire project to GCP.
        """),

        ("Strong Fit", """
Summary
Junior Full-Stack Developer (Co-op) passionate about Node.js/Express, React, and MySQL. Familiar with GCP, CI/CD, and TDD.

Technical Skills
Languages: JavaScript (ES6+), TypeScript, SQL, HTML, CSS
Frameworks/Libraries: Node.js, Express, React, Bootstrap
Databases: MySQL (schema design, indexes, query optimization)
Cloud/DevOps: Google Cloud Platform (Cloud Run/Cloud SQL), GitHub Actions (CI/CD)
Testing: Mocha, Chai
Tools: GitHub, Postman, Figma, Jira
Concepts: OOP, Agile/Scrum

Experience
Full-Stack Developer (Co-op) — Capstone Project | 2025
• Built REST APIs in Node.js/Express and a React SPA with protected routes
• Designed MySQL schema and tuned slow queries with indexes
• Set up GitHub Actions to run Mocha/Chai tests and deploy to GCP Cloud Run
• Wrote docs and collaborated with QA/UX for accessible UI

Education
BSc, Computer Science — 2025
        """),

        ("Borderline (No MySQL)", """
Summary
Junior Full-Stack Developer focused on Node.js and React with CI/CD on GCP. Primary DB experience is PostgreSQL and MongoDB.

Technical Skills
Frameworks/Libraries: Node.js, Express, React, Bootstrap
Databases: PostgreSQL, MongoDB   # intentionally no MySQL
Cloud/DevOps: GCP (Cloud Run), GitHub Actions
Testing: Mocha, Chai
Tools: GitHub, Postman
Concepts: OOP, Agile/Scrum

Experience
Full-Stack Intern — Student Lab | 2025
• Implemented Node.js APIs and React UI
• Deployed via GitHub Actions -> GCP Cloud Run
• Modeled relational schemas in PostgreSQL

Education
Diploma, Computer Systems Technology — 2025
        """),

        ("Off-Target", """
Summary
Web Content Specialist with CMS/SEO focus. Experienced with WordPress, PHP, and marketing automation.

Technical Skills
CMS/Tools: WordPress, HubSpot, Squarespace
Languages: PHP, HTML, CSS, basic JavaScript
Databases: SQLite (basic)
Automation/Analytics: Gulp, Google Analytics/Search Console
Version Control: Git (basic)

Experience
Web Content Specialist — BrightLeaf Media | 2023–2025
• Built and maintained WordPress sites and landing pages
• Improved SEO traffic by 35% YoY
• Automated newsletter workflows via HubSpot

Education
BA, Communications — 2021
        """),
    ]

    # Expected score ranges (tune after first real run if needed)
    expected_scores = {
        "Provided Resume": (60, 100),
        "Strong Fit": (80, 100),
        "Borderline (No MySQL)": (0, 40),
        "Off-Target": (0, 10),
    }

    for title, resume_text in resumes:
        print(f"\n=== Running Test: {title} ===")
        result = analyze_resume_with_job(resume_text, job_description)

        # OG-style assertion using a min/max tuple
        min_score, max_score = expected_scores[title]
        score = result["match_percentage"]

        # Print like your OG script
        print(f"Score: {score}")
        print(f"Must-Haves: {result['must_have_results']}")
        print(f"Nice-to-Haves: {result['nice_have_results']}")

        assert min_score <= score <= max_score, (
            f"Test failed! Got score {score} for:\n"
            f"Must-Haves: {result['must_have_results']}\n"
            f"Nice-to-Haves: {result['nice_have_results']}\n"
        )