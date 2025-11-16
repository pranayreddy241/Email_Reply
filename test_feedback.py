"""
test_feedback.py — run local simulations of feedback to validate sentiment,
discount logic, GPT fallbacks, and reply generation.
"""

from agent_plus_fixed import (
    analyze_sentiment_with_backoff,
    choose_discount,
    generate_personalized_reply,
    _random_code
)

TEST_CASES = [
    "Everything was amazing! The food was perfect and I loved the service.",
    "Service was good, nothing special but okay.",
    "The food was cold and the waiter was rude.",
    "Terrible experience. Worst meal ever. I will never return.",
    "Food was okay but we waited 40 minutes.",
    "Loved the chicken! Delicious and perfect.",
    "Horrible, disgusting food. Refund me.",
    "It was fine, I guess. Nothing to say.",
    "Thank you for the wonderful service!",
    "This was unacceptable. Furious about how long it took.",
    "Great food, friendly staff!",
    "It was disappointing overall.",
    "The portion was small and cold.",
    "Amazing dessert but long wait.",
    "Worst service of my life. Never again.",
    "Everything was perfect!",
    "Bad. Very bad. Cold food.",
    "We enjoyed our dinner a lot!",
    "Not good, not terrible, just average.",
    "Unacceptable experience."
]


def run_test(message, customer_name="John"):
    print("=" * 70)
    print("CUSTOMER MESSAGE:")
    print(message)
    print("-" * 70)

    # 1) sentiment + upset score
    sentiment, score = analyze_sentiment_with_backoff(message)
    print(f"Sentiment → {sentiment}, Score → {score}")

    # 2) discount selection
    discount = choose_discount(sentiment, score, message)
    print(f"Assigned Discount → {discount}%")

    # 3) coupon code
    code = _random_code("TEST", discount)
    print(f"Coupon Code → {code}")

    # 4) reply (GPT or fallback)
    reply = generate_personalized_reply(customer_name, sentiment, score, discount, code, message)
    print("\n=== GENERATED REPLY ===")
    print(reply)
    print()


if __name__ == "__main__":
    for msg in TEST_CASES:
        run_test(msg)
