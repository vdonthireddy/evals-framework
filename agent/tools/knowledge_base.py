"""Knowledge base search tool with simulated company documents."""

from __future__ import annotations

from typing import Any

from agent.tools.base import BaseTool, ToolResult

# Simulated knowledge base articles
_KNOWLEDGE_BASE: list[dict[str, str]] = [
    # Policies
    {"id": "POL-001", "title": "Employee Vacation Policy", "category": "policies", "content": "All full-time employees are entitled to 20 days of paid vacation per year. Vacation days must be requested at least 2 weeks in advance through the HR portal. Unused vacation days can be carried over up to a maximum of 5 days into the next calendar year."},
    {"id": "POL-002", "title": "Remote Work Policy", "category": "policies", "content": "Employees may work remotely up to 3 days per week with manager approval. A stable internet connection and a dedicated workspace are required. Employees must be available during core hours (10am-4pm) in their local timezone."},
    {"id": "POL-003", "title": "Expense Reimbursement Policy", "category": "policies", "content": "Business expenses must be submitted within 30 days of incurrence. Receipts are required for all expenses over $25. Pre-approval is required for expenses exceeding $500. Reimbursement is processed within 10 business days."},
    {"id": "POL-004", "title": "Code of Conduct", "category": "policies", "content": "All employees must adhere to the company code of conduct. This includes treating colleagues with respect, maintaining confidentiality of proprietary information, avoiding conflicts of interest, and reporting any ethical concerns through the anonymous hotline."},
    {"id": "POL-005", "title": "Data Security Policy", "category": "policies", "content": "All company data must be stored on approved systems. Personal devices must have encryption enabled. Passwords must be at least 12 characters with MFA enabled. Sharing credentials is strictly prohibited. Report any security incidents immediately."},
    {"id": "POL-006", "title": "Refund Policy", "category": "policies", "content": "Customers may request a full refund within 30 days of purchase for any reason. After 30 days, refunds are prorated based on usage. Enterprise customers should contact their account manager. Refunds are processed within 5-7 business days to the original payment method."},
    {"id": "POL-007", "title": "Parental Leave Policy", "category": "policies", "content": "Primary caregivers receive 16 weeks of fully paid parental leave. Secondary caregivers receive 8 weeks of fully paid leave. Leave can be taken continuously or in blocks within the first year. Benefits continue during parental leave."},

    # Products
    {"id": "PRD-001", "title": "DataSync Pro - Product Overview", "category": "products", "content": "DataSync Pro is our flagship data integration platform. It supports 200+ connectors, real-time and batch synchronization, automatic schema detection, and built-in data quality monitoring. Pricing starts at $499/month for up to 1M records."},
    {"id": "PRD-002", "title": "DataSync Pro - API Documentation", "category": "products", "content": "The DataSync Pro REST API allows programmatic management of connections, syncs, and monitoring. Base URL: https://api.datasyncpro.example.com/v2. Authentication: Bearer token via API key. Rate limits: 1000 requests per minute per key."},
    {"id": "PRD-003", "title": "CloudGuard - Security Suite", "category": "products", "content": "CloudGuard provides comprehensive cloud security monitoring. Features include threat detection, compliance reporting (SOC2, HIPAA, GDPR), vulnerability scanning, and incident response automation. Integrates with AWS, Azure, and GCP."},
    {"id": "PRD-004", "title": "Analytics Dashboard - User Guide", "category": "products", "content": "The Analytics Dashboard provides real-time insights into your data pipelines. Create custom dashboards with drag-and-drop widgets. Set up alerts for anomalies. Export reports in PDF, CSV, and Excel formats. Supports collaborative editing."},
    {"id": "PRD-005", "title": "Mobile App - Feature Guide", "category": "products", "content": "Our mobile app is available on iOS and Android. Features include push notifications for pipeline alerts, quick status checks, approval workflows, and team chat. Requires DataSync Pro subscription."},

    # FAQs
    {"id": "FAQ-001", "title": "How do I reset my password?", "category": "faq", "content": "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and you'll receive a reset link within 5 minutes. If you don't receive the email, check your spam folder or contact support@example.com."},
    {"id": "FAQ-002", "title": "What are the system requirements?", "category": "faq", "content": "DataSync Pro requires: Chrome 90+, Firefox 88+, or Safari 14+. Minimum 4GB RAM recommended. API integrations require network access to port 443. On-premise deployments require Docker 20+ and Kubernetes 1.24+."},
    {"id": "FAQ-003", "title": "How do I contact support?", "category": "faq", "content": "Contact our support team via email at support@example.com, live chat on our website (24/7), or phone at 1-800-EXAMPLE (business hours: 9am-6pm EST). Enterprise customers have a dedicated Slack channel and a named support engineer."},
    {"id": "FAQ-004", "title": "How do I upgrade my plan?", "category": "faq", "content": "Log in to your account, go to Settings > Billing > Plan. Select the new plan and confirm. Upgrades are prorated. Downgrades take effect at the end of the current billing cycle. Enterprise plan changes require contacting sales."},
    {"id": "FAQ-005", "title": "What integrations are available?", "category": "faq", "content": "DataSync Pro integrates with: Databases (PostgreSQL, MySQL, MongoDB, BigQuery), SaaS (Salesforce, HubSpot, Stripe, Shopify), File Storage (S3, GCS, Azure Blob), and Messaging (Kafka, RabbitMQ, SQS). Custom connectors can be built via our SDK."},
    {"id": "FAQ-006", "title": "Is there a free trial?", "category": "faq", "content": "Yes! We offer a 14-day free trial with full access to all features. No credit card required. After the trial, you can choose a plan or your account will be downgraded to the free tier (limited to 10,000 records/month)."},
    {"id": "FAQ-007", "title": "How is data encrypted?", "category": "faq", "content": "All data is encrypted in transit using TLS 1.3 and at rest using AES-256. Encryption keys are managed via AWS KMS with automatic rotation. Customers on Enterprise plans can bring their own keys (BYOK). SOC2 Type II certified."},

    # Additional diverse articles
    {"id": "POL-008", "title": "Travel Policy", "category": "policies", "content": "Business travel must be pre-approved by your manager. Book flights in economy class for trips under 6 hours. Hotel stays should not exceed $250/night. Per diem for meals is $75/day. Use the corporate travel portal for all bookings."},
    {"id": "PRD-006", "title": "DataSync Pro - Pricing Tiers", "category": "products", "content": "Starter: $499/month (1M records, 10 connectors). Professional: $1,499/month (10M records, 50 connectors, priority support). Enterprise: Custom pricing (unlimited records, all connectors, dedicated support, SLA guarantees, on-premise option)."},
    {"id": "FAQ-008", "title": "Can I cancel my subscription?", "category": "faq", "content": "Yes, you can cancel anytime from Settings > Billing > Cancel Subscription. Your data will be available for export for 30 days after cancellation. After 30 days, all data is permanently deleted. No cancellation fees apply."},
    {"id": "POL-009", "title": "Performance Review Policy", "category": "policies", "content": "Performance reviews are conducted bi-annually in June and December. Reviews include self-assessment, peer feedback, and manager evaluation. Ratings are on a 1-5 scale. Performance directly impacts bonus and promotion eligibility."},
    {"id": "FAQ-009", "title": "What is the SLA uptime guarantee?", "category": "faq", "content": "Starter and Professional plans: 99.9% uptime SLA. Enterprise plans: 99.99% uptime SLA with financial credits for any downtime. Current uptime status is available at status.example.com. We perform maintenance during Sunday 2-4am UTC."},
    {"id": "PRD-007", "title": "AI Assistant - Beta Feature", "category": "products", "content": "Our new AI Assistant (beta) can help you set up data pipelines using natural language. Simply describe what you want to sync and the AI will configure the connection, mapping, and schedule. Available to Professional and Enterprise customers."},
]


def _keyword_search(
    query: str,
    category: str | None = None,
    max_results: int = 3,
) -> list[dict[str, Any]]:
    """Simple keyword-based search over the knowledge base."""
    query_words = set(query.lower().split())
    scored: list[tuple[float, dict[str, str]]] = []

    for article in _KNOWLEDGE_BASE:
        if category and article["category"] != category:
            continue

        # Score based on keyword overlap with title and content
        title_words = set(article["title"].lower().split())
        content_words = set(article["content"].lower().split())

        title_matches = len(query_words & title_words)
        content_matches = len(query_words & content_words)

        # Title matches are weighted more heavily
        score = (title_matches * 3) + content_matches
        if score > 0:
            scored.append((score, article))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, article in scored[:max_results]:
        results.append({
            "id": article["id"],
            "title": article["title"],
            "category": article["category"],
            "content": article["content"],
            "relevance_score": round(min(score / 10, 1.0), 2),
        })
    return results


class KnowledgeBaseTool(BaseTool):
    """Search the internal knowledge base of company policies, products, and FAQs."""

    @property
    def name(self) -> str:
        return "knowledge_base_search"

    @property
    def description(self) -> str:
        return (
            "Search an internal knowledge base of company policies, "
            "product documentation, and FAQs."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "category": {
                    "type": "string",
                    "enum": ["policies", "products", "faq"],
                    "description": "Optional category filter.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs.get("query", "")
        category: str | None = kwargs.get("category")

        if not query.strip():
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="Query cannot be empty.",
            )

        results = _keyword_search(query, category)

        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                output={
                    "results": [],
                    "message": f"No articles found matching '{query}'.",
                },
            )

        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"results": results, "total": len(results)},
        )
