# Simple CRUD Bulletin Board PRD

## Project Overview
A simple web-based bulletin board application with basic CRUD (Create, Read, Update, Delete) functionality.

## Tech Stack
- **Backend**: Python with FastAPI
- **Database**: SQLite (for simplicity)
- **Frontend**: Simple HTML/CSS/JavaScript (vanilla)
- **ORM**: SQLAlchemy

## Core Features

### 1. Post Management
- **Create Post**: Users can create new posts with title and content
- **List Posts**: Display all posts in a paginated list (10 posts per page)
- **View Post**: View individual post details
- **Update Post**: Edit existing posts (title and content)
- **Delete Post**: Remove posts from the system

### 2. Data Model
```
Post:
- id: Integer (Primary Key, Auto-increment)
- title: String (max 200 chars, required)
- content: Text (required)
- author: String (max 100 chars, required)
- created_at: DateTime (auto-generated)
- updated_at: DateTime (auto-updated)
```

### 3. API Endpoints
- `GET /posts` - List all posts (with pagination)
- `GET /posts/{id}` - Get single post
- `POST /posts` - Create new post
- `PUT /posts/{id}` - Update existing post
- `DELETE /posts/{id}` - Delete post

### 4. Frontend Pages
- **Home Page** (`/`): List of posts with pagination
- **Create Post Page** (`/create`): Form to create new post
- **View Post Page** (`/posts/{id}`): Display single post
- **Edit Post Page** (`/posts/{id}/edit`): Form to edit existing post

## Non-Functional Requirements
- Simple and clean UI
- Basic error handling
- Input validation (title and content required)
- Responsive design for mobile/desktop
- No authentication required (public board)

## Project Structure
```
bulletin-board/
├── backend/
│   ├── main.py          # FastAPI application
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas
│   ├── database.py      # Database configuration
│   └── crud.py          # CRUD operations
├── frontend/
│   ├── index.html       # Home page
│   ├── create.html      # Create post page
│   ├── view.html        # View post page
│   ├── edit.html        # Edit post page
│   ├── style.css        # Styling
│   └── script.js        # JavaScript functionality
├── requirements.txt     # Python dependencies
└── README.md           # Project documentation
```

## Success Criteria
1. Users can successfully create, read, update, and delete posts
2. All API endpoints return appropriate status codes
3. Frontend properly displays data and handles errors
4. Data persists in SQLite database
5. Basic input validation works correctly