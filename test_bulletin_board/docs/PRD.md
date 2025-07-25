# Simple CRUD Bulletin Board - Product Requirements Document

## Project Overview
A simple web-based bulletin board system that allows users to create, read, update, and delete posts. This is a basic CRUD application designed to demonstrate fundamental web development concepts.

## Technical Requirements

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (for simplicity)
- **ORM**: SQLAlchemy
- **API Type**: RESTful API with JSON responses

### Frontend
- **Technologies**: HTML5, CSS3, Vanilla JavaScript
- **Design**: Simple, responsive, and clean interface
- **No external frontend frameworks** (to keep it simple)

## Functional Requirements

### 1. Post Model
Each post should have:
- `id`: Unique identifier (auto-incrementing integer)
- `title`: Post title (string, max 200 characters, required)
- `content`: Post content (text, required)
- `author`: Author name (string, max 100 characters, required)
- `created_at`: Creation timestamp (datetime, auto-generated)
- `updated_at`: Last update timestamp (datetime, auto-updated)

### 2. API Endpoints
The backend should provide the following RESTful endpoints:

- `GET /api/posts` - List all posts (sorted by created_at descending)
- `GET /api/posts/{id}` - Get a specific post by ID
- `POST /api/posts` - Create a new post
- `PUT /api/posts/{id}` - Update an existing post
- `DELETE /api/posts/{id}` - Delete a post

### 3. Frontend Features

#### Main Page (index.html)
- Display list of all posts with title, author, and creation date
- "New Post" button to create a new post
- Click on any post to view details
- Pagination (show 10 posts per page)

#### Post Detail View
- Show full post content
- Display author and timestamps
- "Edit" button to modify the post
- "Delete" button to remove the post
- "Back to List" button

#### Create/Edit Post Form
- Form fields for title, content, and author
- "Save" button to submit
- "Cancel" button to return without saving
- Client-side validation for required fields

### 4. User Interface Requirements
- Clean and modern design
- Responsive layout (mobile-friendly)
- Consistent color scheme
- Loading indicators for API calls
- Success/error messages for user actions

### 5. Error Handling
- Proper HTTP status codes
- User-friendly error messages
- Handle network errors gracefully
- Validate all user inputs

## Project Structure
```
test_bulletin_board/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── database.py      # Database configuration
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── schemas.py       # Pydantic schemas
│   │   └── crud.py          # CRUD operations
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Main page
│   ├── style.css            # Styling
│   └── script.js            # JavaScript functionality
└── README.md                # Setup and usage instructions
```

## Development Guidelines
1. Keep the code simple and well-commented
2. Follow Python PEP 8 style guide
3. Use semantic HTML5 elements
4. Implement proper CORS handling
5. Include basic security measures (input sanitization)

## Testing Requirements
- Manual testing of all CRUD operations
- Verify responsive design on different screen sizes
- Test error scenarios (invalid inputs, network errors)
- Ensure proper data persistence

## Deliverables
1. Fully functional backend API
2. Complete frontend application
3. README with setup instructions
4. Sample data for initial testing

## Success Criteria
- All CRUD operations work correctly
- Clean and intuitive user interface
- Proper error handling and user feedback
- Code is well-organized and documented
- Application runs without errors