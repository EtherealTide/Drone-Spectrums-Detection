BUTTON_STYLE = """
QPushButton {
    border: none;
    padding: 5px 10px;
    font-family: 'Segoe UI';
    font-size: 16px;
    color: white;
    text-align: center;
    text-decoration: none;
    margin: 4px 2px;
    border-radius: 4px;
}
QPushButton:hover {
    background-color: #45a049;
}
QPushButton:pressed {
    background-color: #3e8e41;
}
"""
# 浠庝笂鍒颁笅鎵€鏈夊弬鏁拌В閲婏細
# border: none;               # 鏃犺竟妗?
# padding: 5px 10px;         # 鍐呰竟璺濓紝涓婁笅5px锛屽乏鍙?0px
# font-family: 'Segoe UI';   # 瀛椾綋
# font-size: 16px;           # 瀛椾綋澶у皬
# color: white;              # 瀛椾綋棰滆壊
# text-align: center;        # 鏂囧瓧灞呬腑
# text-decoration: none;     # 鏃犱笅鍒掔嚎
# margin: 4px 2px;          # 澶栬竟璺濓紝涓婁笅4px锛屽乏鍙?px
# border-radius: 4px;       # 杈规鍦嗚4px
# 榧犳爣鎮仠鏃惰儗鏅鑹插彉涓?45a049
# 榧犳爣鎸変笅鏃惰儗鏅鑹插彉涓?3e8e41
ADD_BUTTON_STYLE = (
    BUTTON_STYLE
    + """
QPushButton {
    background-color: #4CAF50;
}
QPushButton:hover {
    background-color: #45a049;
}
QPushButton:pressed {
    background-color: #3e8e41;
}
"""
)
CONFIRM_BUTTON_STYLE = ADD_BUTTON_STYLE
