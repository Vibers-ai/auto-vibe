# Fortune Card Web Service PRD - Minimum Viable Product (MVP) Version

## 1. Overview

This document defines the implementation requirements for a web service that offers a core set of features: user registration, login, drawing fortune cards, and viewing past records.

---

## 2. Technology Stack

* **Frontend:** **React (Next.js)**
    * **Rendering:** Leverages Next.js's Server-Side Rendering (SSR) or Static Site Generation (SSG) for optimized initial loading performance and SEO.
    * **State Management:** Utilizes React Context API or a lightweight state management library.
    * **Styling:** Employs **Tailwind CSS for styling, with a UI composed primarily of blue tones and black (dark mode theme).**
* **Backend:** **Node.js (Express)**
    * **API Development:** Implements necessary endpoints following RESTful API principles.
    * **Authentication/Authorization:** Builds a JWT-based authentication system.
* **Database:** **MySQL**
    * **Schema:** Stores user information (ID, email, hashed password, nickname) and card draw records (record ID, user ID, date, question, drawn card information, interpretation text).

---

## 3. Core Features and Page Implementation Details

### 3.1. User and Authentication

#### 3.1.1. Registration Page

* **UI/UX:** Features input fields for email, password, password confirmation, nickname, along with a "Sign Up" button and a link to the login page. **It applies a dark mode style with blue accent elements and a black background.**
* **Frontend Implementation:**
    * Performs **client-side validation** for each input field (email format, password length, match, nickname uniqueness) based on input changes.
    * Sends user data (email, password, nickname) via a **`POST /api/register`** API call.
* **Backend Implementation:**
    * Implements the `POST /api/register` endpoint.
    * Performs **server-side validation** and checks for duplicate emails/nicknames in the database.
    * **Hashes passwords using Bcrypt** before storing them in the database.
    * Returns a 201 Created response on success, and appropriate error responses (e.g., 409 Conflict) on failure.

#### 3.1.2. Login Page

* **UI/UX:** Provides input fields for email and password, a "Login" button, and a link to the registration page. **It maintains a consistent blue and black UI theme, matching the registration page.**
* **Frontend Implementation:**
    * Sends email and password via a **`POST /api/login`** API call.
    * On success, securely **stores the JWT received from the backend in local storage**.
    * Displays an error message on failure.
* **Backend Implementation:**
    * Implements the `POST /api/login` endpoint.
    * Retrieves user from the database and **compares the hashed password**.
    * On successful authentication, **generates and returns a JWT**.
    * Returns a 401 Unauthorized response on failure.

### 3.2. Fortune Card Service

#### 3.2.1. Fortune Card Draw Page

* **UI/UX:** Includes an "Draw Today's Fortune Card" button, a simple question input modal/popup, a card drawing animation, and a result display area. **The overall UI features a dark background (black) with blue-toned UI elements (buttons, selection areas) emphasized to provide a mystical and focused user experience.**
* **Frontend Implementation:**
    * **"Draw Today's Fortune Card" button:**
        * Initiates a **card drawing animation** (e.g., card shuffling, selecting a card from the deck) upon click.
        * After the animation completes, calls the **`GET /api/luck-card/daily`** API to receive today's fortune card and interpretation data.
        * Displays the card image and a brief message, **highlighted with blue font or background.**
    * **"My Question Fortune Card" feature:**
        * Clicking the button displays a **question input modal/popup** (the modal/popup also adheres to the dark theme).
        * After the user enters a question and confirms, sends the question content via a **`POST /api/luck-card/draw`** API call.
        * After the API response, initiates a **card drawing animation** (allowing the user to click to select 1 or more cards).
        * Renders the result based on the card information and interpretation received from the backend.
    * **Card Drawing Interface:** Implements card shuffling and flipping effects using CSS animations or JavaScript libraries. Adds interactive elements to allow users to click and select cards. The card designs themselves should also complement the blue and black theme.
    * **Fortune Card Result Display:** Shows the drawn card image, card name, brief interpretation text, and comprehensive interpretation text. **Text is rendered in bright blue or white for readability, with important parts emphasized.**
    * **Save Result Button:** Only displayed for logged-in users. Clicking it calls the **`POST /api/luck-card/save`** API to save the current fortune card result.
* **Backend Implementation:**
    * **`GET /api/luck-card/daily` endpoint:**
        * Upon request, the server **randomly selects 1 fortune card** and returns its image URL and a message.
    * **`POST /api/luck-card/draw` endpoint:**
        * Receives the question content from the request body.
        * Randomly selects 1 or more fortune cards.
        * **Generates comprehensive interpretation text** based on the selected cards and question, then returns it.
    * **`POST /api/luck-card/save` endpoint:**
        * **Authenticates the user via JWT**.
        * Receives card result information (question, list of drawn cards, interpretation text, date) from the request body.
        * Stores this information in the card history table in the database.

### 3.3. View Records

#### 3.3.1. My Page (My Records)

* **UI/UX:** Displays a list of fortune card draw records (date, question, representative card image). Clicking each record shows detailed content. **This page also uses a black background with blue accent elements and white or light blue text, providing a consistent dark mode experience.**
* **Frontend Implementation:**
    * Upon page load, calls the **`GET /api/user/luck-card-history`** API to retrieve the current logged-in user's fortune card record list.
    * **Renders the record list** based on the received data. Each record item is displayed in a clean card format, with potential blue highlight effects on hover.
    * Clicking each record item calls the **`GET /api/luck-card/:cardHistoryId`** API (including the `cardHistoryId`) to retrieve and display detailed records (reusing the fortune card draw page's result display UI).
* **Backend Implementation:**
    * **`GET /api/user/luck-card-history` endpoint:**
        * **Authenticates the user via JWT**.
        * Retrieves all fortune card records for the authenticated user ID, sorted by most recent, and returns them.
    * **`GET /api/luck-card/:cardHistoryId` endpoint:**
        * **Authenticates the user via JWT**.
        * Retrieves detailed fortune card record information from the database where `cardHistoryId` matches and the user ID matches the logged-in user.
        * Returns a 403 Forbidden or 404 Not Found response for unauthorized access or non-existent records.