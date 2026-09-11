"""
Tests for betting/promo.py — every number hand-computed.
"""
import pytest

from betting.promo import (apply_slippage, best_free_bet_conversion,
                           effective_decimal, hedge_free_bet,
                           hedge_profit_boost, hedge_risk_free)


class TestVenueCosts:
    def test_book_passthrough(self):
        assert effective_decimal("book", price=-110) == pytest.approx(1 + 10 / 11)

    def test_kalshi_taker_fee(self):
        # q=0.5: fee 0.07*0.25 = 0.0175 -> cost 0.5175 -> dec 1.9324
        assert effective_decimal("kalshi", contract_price=0.5) == \
            pytest.approx(1 / 0.5175)

    def test_polymarket_cheaper_taker(self):
        k = effective_decimal("kalshi", contract_price=0.5)
        p = effective_decimal("polymarket", contract_price=0.5)
        assert p > k

    def test_maker_no_fee(self):
        assert effective_decimal("kalshi_maker", contract_price=0.5) == \
            pytest.approx(2.0)

    def test_bad_contract_price(self):
        with pytest.raises(ValueError):
            effective_decimal("kalshi", contract_price=1.5)


class TestHedgeMath:
    def test_free_bet_hand_computed(self):
        # $100 free bet at +400 (dec 5), hedge at dec 2.0:
        # h = 100*4/2 = 200; guaranteed = 400 - 200 = 200. (A dec-2.0
        # opposite of +400 never coexists at real books — this checks
        # the algebra, the realistic case is below.)
        plan = hedge_free_bet(100, 5.0, 2.0)
        assert plan.hedge_stake == pytest.approx(200.0)
        assert plan.guaranteed == pytest.approx(200.0)
        assert plan.conversion == pytest.approx(2.0)
        # both outcomes really do net the same
        assert plan.bonus_win_pnl + plan.hedge_lose_pnl == \
            pytest.approx(plan.bonus_lose_pnl + plan.hedge_win_pnl)

    def test_free_bet_realistic_conversion(self):
        # +400 bonus leg, hedge side at -450 (dec 1.2222):
        # h = 400/1.2222 = 327.27; guaranteed 72.73 -> 72.7%
        plan = hedge_free_bet(100, 5.0, 1.0 + 100 / 450)
        assert plan.conversion == pytest.approx(0.7273, abs=1e-3)

    def test_risk_free_hand_computed(self):
        # $100 at dec 2.0, refund worth 0.7, hedge at dec 2.0:
        # h = 100*(2-0.7)/2 = 65; win: 100-65=35; lose: -100+70-... :
        # -100 + 65*(1) + 70 = 35. Equal.
        plan = hedge_risk_free(100, 2.0, 2.0, refund_value=0.7)
        assert plan.hedge_stake == pytest.approx(65.0)
        assert plan.guaranteed == pytest.approx(35.0)
        assert plan.bonus_win_pnl + plan.hedge_lose_pnl == \
            pytest.approx(plan.bonus_lose_pnl + plan.hedge_win_pnl)

    def test_profit_boost_clears_or_flags(self):
        # +100 (dec 2) boosted 50% -> d'=2.5; hedge dec 2.0:
        # h = 100*2.5/2 = 125; guaranteed = 150-125 = 25
        plan = hedge_profit_boost(100, 2.0, 0.5, 2.0)
        assert plan.guaranteed == pytest.approx(25.0)
        assert not plan.notes
        # tiny boost at heavy vig: negative lock is flagged
        weak = hedge_profit_boost(100, 2.0, 0.02, 1.85)
        assert weak.guaranteed < 0 and weak.notes

    def test_longshot_maximizes_free_bet_conversion(self):
        cands = [
            ("fav -150", 1 + 100 / 150, {"venue": "book", "price": 130}),
            ("dog +400", 5.0, {"venue": "kalshi", "contract_price": 0.82}),
        ]
        label, plan = best_free_bet_conversion(
            100, cands, lambda **kw: effective_decimal(**kw))
        assert label == "dog +400"
        assert plan.conversion > 0.6

    def test_slippage_widens_guarantee(self):
        plan = hedge_free_bet(100, 5.0, 2.0)
        g = plan.guaranteed
        assert apply_slippage(plan, 0.02).guaranteed == pytest.approx(g * 0.98)
