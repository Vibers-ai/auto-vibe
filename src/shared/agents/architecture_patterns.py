"""Architecture Pattern Library for Master Planner

This module contains predefined architecture patterns that the Master Planner
can use to generate better project architectures based on project type.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class ProjectType(Enum):
    """Common project types."""
    WEB_APP = "web_app"
    MOBILE_APP = "mobile_app"
    API_SERVICE = "api_service"
    DESKTOP_APP = "desktop_app"
    MICROSERVICES = "microservices"
    DATA_PIPELINE = "data_pipeline"
    ML_SYSTEM = "ml_system"
    IOT_SYSTEM = "iot_system"
    BLOCKCHAIN = "blockchain"
    E_COMMERCE = "e_commerce"
    SOCIAL_PLATFORM = "social_platform"
    CMS = "cms"
    SAAS = "saas"


class ArchitecturePattern(Enum):
    """Common architecture patterns."""
    MONOLITHIC = "monolithic"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    MVC = "mvc"
    MVVM = "mvvm"
    CLEAN_ARCH = "clean_architecture"
    HEXAGONAL = "hexagonal"
    CQRS = "cqrs"
    JAMSTACK = "jamstack"


@dataclass
class TechnologyStack:
    """Technology stack definition."""
    frontend: List[str] = field(default_factory=list)
    backend: List[str] = field(default_factory=list)
    database: List[str] = field(default_factory=list)
    cache: List[str] = field(default_factory=list)
    message_queue: List[str] = field(default_factory=list)
    devops: List[str] = field(default_factory=list)
    monitoring: List[str] = field(default_factory=list)
    testing: List[str] = field(default_factory=list)


@dataclass
class ProjectStructure:
    """Recommended project structure."""
    directories: Dict[str, str] = field(default_factory=dict)
    key_files: List[str] = field(default_factory=list)
    naming_conventions: Dict[str, str] = field(default_factory=dict)


@dataclass
class ArchitectureTemplate:
    """Complete architecture template."""
    name: str
    description: str
    pattern: ArchitecturePattern
    tech_stack: TechnologyStack
    structure: ProjectStructure
    best_for: List[str] = field(default_factory=list)
    considerations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern.value,
            "tech_stack": {
                "frontend": self.tech_stack.frontend,
                "backend": self.tech_stack.backend,
                "database": self.tech_stack.database,
                "cache": self.tech_stack.cache,
                "message_queue": self.tech_stack.message_queue,
                "devops": self.tech_stack.devops,
                "monitoring": self.tech_stack.monitoring,
                "testing": self.tech_stack.testing
            },
            "structure": {
                "directories": self.structure.directories,
                "key_files": self.structure.key_files,
                "naming_conventions": self.structure.naming_conventions
            },
            "best_for": self.best_for,
            "considerations": self.considerations
        }


class ArchitecturePatternLibrary:
    """Library of architecture patterns and templates."""
    
    def __init__(self):
        self.templates: Dict[str, ArchitectureTemplate] = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize with common architecture templates."""
        
        # Modern Full-Stack Web Application
        self.templates["modern_fullstack"] = ArchitectureTemplate(
            name="Modern Full-Stack Web Application",
            description="A scalable full-stack web application with React/Next.js frontend and Node.js backend",
            pattern=ArchitecturePattern.LAYERED,
            tech_stack=TechnologyStack(
                frontend=["Next.js 14+", "React 18+", "TypeScript", "Tailwind CSS", "Zustand/Redux Toolkit"],
                backend=["Node.js", "Express/Fastify", "TypeScript", "Prisma ORM"],
                database=["PostgreSQL", "Redis"],
                cache=["Redis", "CDN (Cloudflare/CloudFront)"],
                devops=["Docker", "GitHub Actions", "Vercel/AWS"],
                monitoring=["Sentry", "LogRocket", "Prometheus"],
                testing=["Jest", "React Testing Library", "Cypress", "Supertest"]
            ),
            structure=ProjectStructure(
                directories={
                    "frontend": "Next.js app with src/app structure",
                    "backend": "Express API with layered architecture",
                    "shared": "Shared types and utilities",
                    "infrastructure": "Docker and deployment configs"
                },
                key_files=[
                    "frontend/src/app/layout.tsx",
                    "frontend/src/app/page.tsx",
                    "backend/src/server.ts",
                    "backend/src/routes/index.ts",
                    "docker-compose.yml",
                    ".github/workflows/ci.yml"
                ],
                naming_conventions={
                    "components": "PascalCase (e.g., UserProfile.tsx)",
                    "utilities": "camelCase (e.g., formatDate.ts)",
                    "api_routes": "kebab-case (e.g., /api/user-profile)",
                    "database": "snake_case (e.g., user_profiles)"
                }
            ),
            best_for=["SaaS applications", "E-commerce platforms", "Social platforms", "Content management systems"],
            considerations=[
                "Use Server Components for better performance",
                "Implement proper caching strategies",
                "Set up CI/CD from the start",
                "Use environment variables for configuration",
                "Implement proper error boundaries"
            ]
        )
        
        # Microservices E-commerce Platform
        self.templates["microservices_ecommerce"] = ArchitectureTemplate(
            name="Microservices E-commerce Platform",
            description="A scalable e-commerce platform using microservices architecture",
            pattern=ArchitecturePattern.MICROSERVICES,
            tech_stack=TechnologyStack(
                frontend=["Next.js", "React", "TypeScript", "Tailwind CSS", "React Query"],
                backend=["Node.js (Gateway)", "Python (Services)", "Go (Performance-critical services)"],
                database=["PostgreSQL (Users, Orders)", "MongoDB (Catalog)", "Redis (Sessions, Cache)"],
                message_queue=["RabbitMQ", "Apache Kafka"],
                devops=["Kubernetes", "Docker", "Istio", "ArgoCD"],
                monitoring=["Prometheus", "Grafana", "ELK Stack", "Jaeger"],
                testing=["Jest", "Pytest", "k6 (Load testing)", "Postman"]
            ),
            structure=ProjectStructure(
                directories={
                    "services/user-service": "User management microservice",
                    "services/product-service": "Product catalog microservice",
                    "services/order-service": "Order processing microservice",
                    "services/payment-service": "Payment processing microservice",
                    "services/notification-service": "Email/SMS notifications",
                    "api-gateway": "Kong/Express gateway",
                    "frontend": "Next.js customer frontend",
                    "admin-frontend": "React admin dashboard"
                },
                key_files=[
                    "docker-compose.yml",
                    "kubernetes/deployments/",
                    "api-gateway/src/index.ts",
                    ".gitlab-ci.yml"
                ],
                naming_conventions={
                    "services": "kebab-case (e.g., user-service)",
                    "api_endpoints": "RESTful conventions",
                    "events": "PascalCase (e.g., OrderPlaced)",
                    "database": "service_prefix_table (e.g., user_profiles)"
                }
            ),
            best_for=["Large-scale e-commerce", "Multi-tenant platforms", "High-traffic applications"],
            considerations=[
                "Implement service discovery",
                "Use event sourcing for order tracking",
                "Implement circuit breakers",
                "Set up distributed tracing",
                "Design for eventual consistency"
            ]
        )
        
        # Serverless API Service
        self.templates["serverless_api"] = ArchitectureTemplate(
            name="Serverless API Service",
            description="A cost-effective serverless API using AWS Lambda",
            pattern=ArchitecturePattern.SERVERLESS,
            tech_stack=TechnologyStack(
                backend=["AWS Lambda", "API Gateway", "TypeScript/Python", "Serverless Framework"],
                database=["DynamoDB", "Aurora Serverless"],
                cache=["ElastiCache", "CloudFront"],
                devops=["AWS SAM/Serverless Framework", "GitHub Actions", "AWS CloudFormation"],
                monitoring=["CloudWatch", "X-Ray", "Datadog"],
                testing=["Jest", "AWS SAM Local", "Artillery"]
            ),
            structure=ProjectStructure(
                directories={
                    "functions": "Lambda function handlers",
                    "layers": "Shared code layers",
                    "lib": "Shared libraries",
                    "tests": "Unit and integration tests"
                },
                key_files=[
                    "serverless.yml",
                    "functions/users/handler.ts",
                    "lib/dynamodb.ts",
                    "tests/integration/api.test.ts"
                ],
                naming_conventions={
                    "functions": "camelCase (e.g., getUser)",
                    "files": "kebab-case (e.g., user-handler.ts)",
                    "dynamodb_tables": "PascalCase (e.g., UsersTable)"
                }
            ),
            best_for=["APIs with variable traffic", "Event-driven systems", "Prototypes", "Cost-sensitive projects"],
            considerations=[
                "Design for cold starts",
                "Use Lambda layers for dependencies",
                "Implement proper error handling",
                "Set up structured logging",
                "Monitor Lambda costs"
            ]
        )
        
        # Real-time Collaboration Platform
        self.templates["realtime_collab"] = ArchitectureTemplate(
            name="Real-time Collaboration Platform",
            description="A platform for real-time collaboration with WebSockets",
            pattern=ArchitecturePattern.EVENT_DRIVEN,
            tech_stack=TechnologyStack(
                frontend=["React", "TypeScript", "Socket.io-client", "Tailwind CSS", "Slate.js/Quill"],
                backend=["Node.js", "Socket.io", "Express", "TypeScript"],
                database=["PostgreSQL", "Redis (Pub/Sub)"],
                cache=["Redis"],
                devops=["Docker", "Nginx", "PM2"],
                monitoring=["New Relic", "Sentry"],
                testing=["Jest", "Puppeteer", "Socket.io-client (testing)"]
            ),
            structure=ProjectStructure(
                directories={
                    "client": "React frontend application",
                    "server": "Node.js WebSocket server",
                    "shared": "Shared types and constants",
                    "nginx": "Nginx configuration"
                },
                key_files=[
                    "server/src/socket-handlers/",
                    "client/src/hooks/useSocket.ts",
                    "shared/types/events.ts",
                    "docker-compose.yml"
                ],
                naming_conventions={
                    "events": "SCREAMING_SNAKE_CASE (e.g., USER_JOINED)",
                    "handlers": "camelCase (e.g., handleUserJoined)",
                    "components": "PascalCase"
                }
            ),
            best_for=["Chat applications", "Collaborative editing", "Real-time dashboards", "Online gaming"],
            considerations=[
                "Handle connection failures gracefully",
                "Implement presence system",
                "Use Redis for horizontal scaling",
                "Implement conflict resolution",
                "Add offline support"
            ]
        )
        
        # Machine Learning API System
        self.templates["ml_api_system"] = ArchitectureTemplate(
            name="Machine Learning API System",
            description="A production ML model serving system",
            pattern=ArchitecturePattern.LAYERED,
            tech_stack=TechnologyStack(
                frontend=["React", "TypeScript", "Recharts", "Material-UI"],
                backend=["Python (FastAPI)", "Celery", "Redis", "MLflow"],
                database=["PostgreSQL", "S3 (Model storage)", "Redis (Cache)"],
                message_queue=["RabbitMQ", "Celery"],
                devops=["Docker", "Kubernetes", "Kubeflow", "GitHub Actions"],
                monitoring=["Prometheus", "Grafana", "MLflow", "Weights & Biases"],
                testing=["Pytest", "Locust", "Great Expectations"]
            ),
            structure=ProjectStructure(
                directories={
                    "api": "FastAPI application",
                    "models": "ML model definitions",
                    "training": "Training pipelines",
                    "data": "Data processing scripts",
                    "frontend": "React dashboard"
                },
                key_files=[
                    "api/main.py",
                    "models/model_registry.py",
                    "training/train.py",
                    "docker/Dockerfile.api"
                ],
                naming_conventions={
                    "models": "snake_case (e.g., sentiment_analyzer)",
                    "endpoints": "kebab-case (e.g., /predict-sentiment)",
                    "classes": "PascalCase"
                }
            ),
            best_for=["ML model serving", "Data processing pipelines", "Analytics platforms"],
            considerations=[
                "Version your models",
                "Implement A/B testing",
                "Monitor model drift",
                "Set up data validation",
                "Plan for model updates"
            ]
        )
    
    def get_template(self, name: str) -> Optional[ArchitectureTemplate]:
        """Get a specific architecture template."""
        return self.templates.get(name)
    
    def get_templates_for_project_type(self, project_type: str) -> List[ArchitectureTemplate]:
        """Get recommended templates for a project type."""
        recommendations = []
        
        project_type_lower = project_type.lower()
        
        for template in self.templates.values():
            # Check if any of the "best_for" items match the project type
            for use_case in template.best_for:
                if project_type_lower in use_case.lower() or use_case.lower() in project_type_lower:
                    recommendations.append(template)
                    break
        
        return recommendations
    
    def get_tech_stack_recommendations(self, requirements: List[str]) -> TechnologyStack:
        """Recommend a technology stack based on requirements."""
        stack = TechnologyStack()
        
        requirements_text = " ".join(requirements).lower()
        
        # Frontend recommendations
        if "real-time" in requirements_text or "collaborative" in requirements_text:
            stack.frontend = ["React", "TypeScript", "Socket.io-client"]
        elif "mobile" in requirements_text:
            stack.frontend = ["React Native", "TypeScript"]
        elif "dashboard" in requirements_text or "admin" in requirements_text:
            stack.frontend = ["React", "TypeScript", "Material-UI", "Recharts"]
        else:
            stack.frontend = ["Next.js", "React", "TypeScript", "Tailwind CSS"]
        
        # Backend recommendations
        if "machine learning" in requirements_text or "ml" in requirements_text:
            stack.backend = ["Python", "FastAPI", "Celery"]
        elif "high performance" in requirements_text:
            stack.backend = ["Go", "Gin/Echo"]
        elif "real-time" in requirements_text:
            stack.backend = ["Node.js", "Socket.io", "Express"]
        else:
            stack.backend = ["Node.js", "Express", "TypeScript"]
        
        # Database recommendations
        if "graph" in requirements_text or "social" in requirements_text:
            stack.database = ["Neo4j", "PostgreSQL"]
        elif "document" in requirements_text or "flexible schema" in requirements_text:
            stack.database = ["MongoDB", "Redis"]
        elif "analytics" in requirements_text:
            stack.database = ["PostgreSQL", "ClickHouse", "Redis"]
        else:
            stack.database = ["PostgreSQL", "Redis"]
        
        return stack
    
    def export_templates(self, output_path: str):
        """Export all templates to a JSON file."""
        templates_dict = {
            name: template.to_dict() 
            for name, template in self.templates.items()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(templates_dict, f, indent=2)
    
    def import_templates(self, input_path: str):
        """Import templates from a JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            templates_dict = json.load(f)
        
        # This would need proper deserialization logic
        # For now, keeping it simple
        pass


# Singleton instance
architecture_library = ArchitecturePatternLibrary()