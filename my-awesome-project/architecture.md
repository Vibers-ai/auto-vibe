# System Architecture Document
Generated on: 2025-07-13 00:06:06

## High-Level Architecture

Technology Stack:
- Backend: Node.js with Express.js (integrated into Next.js API Routes)
- Frontend: React with Next.js (using TypeScript)
- Database: MySQL
- Additional Tools:
  - Styling: Tailwind CSS
  - API Client: Axios
  - Backend Libraries: bcryptjs (password hashing), jsonwebtoken (JWT), mysql2 (database driver), joi (validation)
  - Development: ESLint, Prettier

Architecture Pattern: Monolithic Architecture (Next.js App)

Justification:
A monolithic architecture using Next.js is the most efficient choice for this MVP. Next.js's built-in support for API routes allows us to develop and deploy the backend and frontend as a single, cohesive unit. This approach significantly simplifies the development workflow, configuration, and deployment process compared to a microservices architecture, which would be overkill for a project of this scale. It reduces latency between the frontend and backend, enhances code co-location, and streamlines state management and data fetching, aligning perfectly with the project's requirements for rapid development and a tightly integrated user experience.


## Database Schema

-- SQL CREATE TABLE statements for the Fortune Card Web Service

-- Table to store user information
-- Ensures unique emails and nicknames for registration
CREATE TABLE `users` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `email` VARCHAR(255) UNIQUE NOT NULL,
  `password_hash` VARCHAR(255) NOT NULL,
  `nickname` VARCHAR(50) UNIQUE NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table to store the master list of all available fortune cards
-- This table should be pre-populated with card data.
CREATE TABLE `fortune_cards` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `name` VARCHAR(100) NOT NULL,
  `image_url` VARCHAR(255) NOT NULL,
  `brief_interpretation` TEXT NOT NULL,
  `full_interpretation` TEXT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Table to store the history of card draws for each user
CREATE TABLE `draw_history` (
  `id` INT AUTO_INCREMENT PRIMARY KEY,
  `user_id` INT NOT NULL,
  `question` TEXT, -- Can be NULL for daily draws without a specific question
  `generated_interpretation` TEXT NOT NULL,
  `draw_date` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX `idx_user_id_draw_date` (`user_id`, `draw_date` DESC),
  CONSTRAINT `fk_draw_history_user`
    FOREIGN KEY (`user_id`)
    REFERENCES `users`(`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Junction table to link a draw history record with the specific cards that were drawn
-- This supports saving multiple cards for a single draw.
CREATE TABLE `history_cards` (
  `history_id` INT NOT NULL,
  `card_id` INT NOT NULL,
  PRIMARY KEY (`history_id`, `card_id`),
  CONSTRAINT `fk_history_cards_history`
    FOREIGN KEY (`history_id`)
    REFERENCES `draw_history`(`id`)
    ON DELETE CASCADE,
  CONSTRAINT `fk_history_cards_card`
    FOREIGN KEY (`card_id`)
    REFERENCES `fortune_cards`(`id`)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


## API Design

# OpenAPI 3.0 specification for the Fortune Card Web Service
openapi: 3.0.3
info:
  title: Fortune Card Web Service API
  description: API for user authentication, card drawing, and viewing history.
  version: 1.0.0
servers:
  - url: /api
    description: Local development server

paths:
  /register:
    post:
      summary: Register a new user
      tags: [User & Auth]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserRegister'
      responses:
        '201':
          description: User created successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  message:
                    type: string
                    example: User registered successfully.
        '400':
          description: Invalid input data
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
        '409':
          description: Email or nickname already exists
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /login:
    post:
      summary: Log in a user
      tags: [User & Auth]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserLogin'
      responses:
        '200':
          description: Login successful, returns JWT token
          content:
            application/json:
              schema:
                type: object
                properties:
                  token:
                    type: string
                    example: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
        '401':
          description: Unauthorized, invalid credentials
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /luck-card/daily:
    get:
      summary: Get a random daily fortune card
      tags: [Fortune Card]
      responses:
        '200':
          description: A single random fortune card
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Card'
        '500':
          description: Server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /luck-card/draw:
    post:
      summary: Draw a card for a specific question
      tags: [Fortune Card]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [question]
              properties:
                question:
                  type: string
                  example: "Will I succeed in my new project?"
      responses:
        '200':
          description: Drawn card(s) and interpretation
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/DrawResult'
        '400':
          description: Invalid input
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

  /luck-card/save:
    post:
      summary: Save a fortune card draw result
      tags: [Fortune Card]
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SaveDrawRequest'
      responses:
        '201':
          description: Draw result saved successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  historyId:
                    type: integer
                    example: 101
                  message:
                    type: string
                    example: "Draw result saved successfully."
        '401':
          description: Unauthorized
        '400':
          description: Invalid input data

  /user/luck-card-history:
    get:
      summary: Get the logged-in user's draw history
      tags: [History]
      security:
        - bearerAuth: []
      responses:
        '200':
          description: A list of the user's draw history summaries
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/HistorySummary'
        '401':
          description: Unauthorized

  /luck-card/{cardHistoryId}:
    get:
      summary: Get details of a specific card draw
      tags: [History]
      security:
        - bearerAuth: []
      parameters:
        - name: cardHistoryId
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Detailed information about the draw
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HistoryDetail'
        '401':
          description: Unauthorized
        '403':
          description: Forbidden, user does not own this record
        '404':
          description: Record not found

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  schemas:
    UserRegister:
      type: object
      required: [email, password, nickname]
      properties:
        email:
          type: string
          format: email
        password:
          type: string
          minLength: 8
        nickname:
          type: string
          minLength: 3
    UserLogin:
      type: object
      required: [email, password]
      properties:
        email:
          type: string
          format: email
        password:
          type: string
    Card:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        image_url:
          type: string
          format: uri
        brief_interpretation:
          type: string
        full_interpretation:
          type: string
    DrawResult:
      type: object
      properties:
        cards:
          type: array
          items:
            $ref: '#/components/schemas/Card'
        generated_interpretation:
          type: string
    SaveDrawRequest:
      type: object
      required: [card_ids, generated_interpretation]
      properties:
        question:
          type: string
          nullable: true
        card_ids:
          type: array
          items:
            type: integer
        generated_interpretation:
          type: string
    HistorySummary:
      type: object
      properties:
        id:
          type: integer
        question:
          type: string
          nullable: true
        draw_date:
          type: string
          format: date-time
        representative_card_image:
          type: string
          format: uri
    HistoryDetail:
      type: object
      properties:
        id:
          type: integer
        question:
          type: string
          nullable: true
        draw_date:
          type: string
          format: date-time
        generated_interpretation:
          type: string
        cards:
          type: array
          items:
            $ref: '#/components/schemas/Card'
    Error:
      type: object
      properties:
        message:
          type: string


## Frontend Components

Component Hierarchy:

- App (_app.tsx)
  - purpose: Root component to initialize pages. Wraps all pages with global providers (e.g., AuthProvider, ThemeProvider).
  - state: n/a
  - props: Component, pageProps
  - children:
    - AuthProvider
      - Layout

- AuthProvider (context/AuthContext.tsx)
  - purpose: Manages JWT token, user authentication state, and provides it to all components via React Context.
  - state: `token`, `user`, `isLoggedIn`, `isLoading`
  - props: `children`
  - children: n/a

- Layout (components/Layout.tsx)
  - purpose: Provides consistent page structure, including the main header and footer. Applies the global dark theme.
  - state: n/a
  - props: `children`
  - children:
    - Header
    - main
    - Footer (optional)

- Header (components/Header.tsx)
  - purpose: Displays site navigation. Shows different links based on authentication status (Login/Register vs. My Page/Logout).
  - state: n/a
  - props: n/a (consumes AuthContext)
  - children: n/a

- Card (components/Card.tsx)
  - purpose: A reusable component to display a single fortune card with its image and name.
  - state: n/a
  - props: `name: string`, `imageUrl: string`
  - children: n/a

- AuthForm (components/AuthForm.tsx)
  - purpose: A reusable form for both registration and login to reduce code duplication. Handles input fields, client-side validation, and submission logic.
  - state: `formData`, `errors`, `isLoading`
  - props: `formType: 'login' | 'register'`, `onSubmit: (data) => Promise<void>`
  - children: n/a

- QuestionModal (components/QuestionModal.tsx)
  - purpose: A modal dialog that prompts the user to enter a question before drawing a card.
  - state: `questionText`
  - props: `isOpen: boolean`, `onClose: () => void`, `onSubmit: (question) => void`
  - children: n/a

- CardResult (components/CardResult.tsx)
  - purpose: Displays the result of a card draw, including card images, names, and the generated interpretation. Includes a "Save" button for logged-in users.
  - state: n/a
  - props: `cards: Card[]`, `interpretation: string`, `onSave: () => void`, `isLoggedIn: boolean`
  - children:
    - Card (multiple instances)

- Page: / (pages/index.tsx - Card Draw Page)
  - purpose: The main page for drawing fortune cards.
  - state: `drawResult`, `isLoading`, `error`, `isModalOpen`
  - props: n/a
  - children:
    - QuestionModal
    - CardResult

- Page: /register (pages/register.tsx)
  - purpose: Renders the registration form.
  - state: n/a
  - props: n/a
  - children:
    - AuthForm

- Page: /login (pages/login.tsx)
  - purpose: Renders the login form.
  - state: n/a
  - props: n/a
  - children:
    - AuthForm

- Page: /my-page (pages/my-page/index.tsx)
  - purpose: Displays a list of the user's past fortune card draws. Requires authentication.
  - state: `history: HistorySummary[]`, `isLoading`, `error`
  - props: n/a
  - children:
    - HistoryListItem (multiple instances)

- HistoryListItem (components/HistoryListItem.tsx)
  - purpose: Displays a summary of a single draw history record in a list. Links to the detail page.
  - state: n/a
  - props: `item: HistorySummary`
  - children: n/a

- Page: /my-page/[historyId] (pages/my-page/[historyId].tsx)
  - purpose: Displays the full details of a specific past draw.
  - state: `historyDetail: HistoryDetail`, `isLoading`, `error`
  - props: n/a
  - children:
    - CardResult (reused to display the saved draw)


---

This document provides the technical architecture for the project implementation.
Refer to tasks.json for the detailed implementation plan.
