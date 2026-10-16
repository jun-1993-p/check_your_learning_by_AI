"""quiz_templates의 빈칸 문제 만들기(make_question) 테스트. LLM 없이 돈다.

실행: 프로젝트 루트에서 python -m unittest
"""

import unittest

from quiz_templates import BLANK, BlankError, filled, make_question


class MakeQuestionTest(unittest.TestCase):
    # ---------------------------------------------------------------- 기존 동작

    def test_빈칸이_없으면_문장에서_정답을_찾아_가린다(self):
        question, count = make_question("strip 함수는 양쪽 공백을 지운다.", "strip")
        self.assertEqual(question, "____ 함수는 양쪽 공백을 지운다.")
        self.assertEqual(count, 1)

    def test_빈칸_하나에_정답이_또_나오면_그_자리도_가린다(self):
        question, count = make_question(
            "____ 함수를 쓰면 strip 결과가 새 문자열로 나온다.", "strip"
        )
        self.assertEqual(question, "____ 함수를 쓰면 ____ 결과가 새 문자열로 나온다.")
        self.assertEqual(count, 2)

    def test_정답이_비어_있으면_오류(self):
        with self.assertRaises(BlankError):
            make_question("____ 함수는 공백을 지운다.", " ")

    def test_정답이_너무_길면_오류(self):
        with self.assertRaises(BlankError):
            make_question("____ 함수는 공백을 지운다.", "가" * 31)

    def test_빈칸_말고_내용이_없으면_오류(self):
        with self.assertRaises(BlankError):
            make_question(" ____ ", "strip")

    # ---------------------------------------------------------------- 긴 밑줄 정규화

    def test_긴_밑줄은_빈칸_하나로_센다(self):
        question, count = make_question("strip은 양쪽 공백을 ________.", "지운다")
        self.assertEqual(question, f"strip은 양쪽 공백을 {BLANK}.")
        self.assertEqual(count, 1)
        self.assertEqual(filled(question, "지운다"), "strip은 양쪽 공백을 지운다.")

    def test_밑줄_세_개도_빈칸이다(self):
        question, count = make_question("___ 함수는 양쪽 공백을 지운다.", "strip")
        self.assertEqual(question, f"{BLANK} 함수는 양쪽 공백을 지운다.")
        self.assertEqual(count, 1)

    def test_파이썬_이름의_밑줄은_빈칸이_아니다(self):
        text = "__init__ 메서드는 snake_case 이름의 ____ 변수를 만든다."
        question, count = make_question(text, "인스턴스")
        self.assertEqual(question, text)
        self.assertEqual(count, 1)

    def test_이름_안에_이어진_밑줄_세_개_이상도_빈칸이_아니다(self):
        for text in (
            "___private 변수는 ____ 로 표시한다.",
            "a___b 는 ____ 이다.",
            "__init___ 는 ____ 이다.",
        ):
            with self.subTest(text=text):
                question, count = make_question(text, "관례")
                self.assertEqual(question, text)
                self.assertEqual(count, 1)

    # ---------------------------------------------------------------- LLM이 쓴 빈칸 개수

    def test_LLM이_빈칸을_여러_개_쓰면_오류(self):
        text = "딕셔너리 ____에서 '____'의 Value를 얻기 위해 ____['____']을 사용한다."
        with self.assertRaisesRegex(BlankError, "4"):
            make_question(text, "a")

    def test_긴_밑줄_두_덩어리도_빈칸_두_개라_오류(self):
        with self.assertRaisesRegex(BlankError, "2"):
            make_question("________ 함수는 공백을 ______.", "strip")


if __name__ == "__main__":
    unittest.main()
