#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/22 18:32
# @Author  : xiaorun
# @Site    : 
# @File    : account.py
# @Software: PyCharm
from wtforms import Form, StringField, IntegerField, PasswordField, SubmitField
from wtforms.validators import Length, NumberRange, DataRequired, Email, ValidationError, EqualTo
from app.models.user import User
from flask_wtf import FlaskForm


# 这里使用wtforms类来创建注册的类
class RegisterForm(Form):
    email = StringField(
        validators=[
            DataRequired(),
            Length(8, 64),
            Email(message='电子邮箱不符合规范')
        ],
        render_kw={
            'placeholder': '请输入电子邮件！'
        }
    )
    password = PasswordField(
        validators=[
            DataRequired(message='密码不可以为空，请输入你的密码'),
            Length(6, 32, message='密码最小6位最长32')
        ],
        render_kw={
            'placeholder': '请输入密码！'
        }
    )
    password2 = PasswordField(
        validators=[
            DataRequired(message='密码不能为空，请输入你的密码'),
            Length(6, 32, message='密码最小6位最长32'),
            EqualTo('password', message="两次密码不一致!")
        ],
        render_kw={
            'placeholder': '请确认密码！'
        }
    )
    nickname = StringField(
        validators=[
            DataRequired(),
            Length(2, 10, message='昵称至少需要2个字符，最多10个字符')
        ],
        render_kw={
            'placeholder': '请输入昵称！'
        }
    )
    submit = SubmitField(
        '提交'
    )

    # 判断数据库中是否有同名的email
    # 这里的field是wtform自动传入的，是客户端浏览器传入的参数
    # wtform 会自动检测 validate_email这个函数将这个函数和email属性做关联
    def validate_email(self, field):
        # 不论查询出多少条，都只返回1条
        if User.query.filter_by(email=field.data).first():
            raise ValidationError("电子邮件已被注册")

    def validate_nickname(self, field):
        if User.query.filter_by(nickname=field.data).first():
            raise ValidationError('昵称已存在')


# 这里使用Flask-WTF包来创建登录验证的类
# class LoginForm(Form):
class LoginForm(FlaskForm):
    email = StringField(
        validators=[
            DataRequired(),
            Length(8, 64),
            Email(message='电子邮箱不符合规范')
        ],
        render_kw={
            'placeholder': '请输入注册邮箱！'
        }
    )
    password = PasswordField(
        validators=[
            DataRequired(message='密码不可以为空，请输入你的密码'),
            Length(6, 32, message='密码长度是6到32位')
        ],
        render_kw={
            'placeholder': '请输入密码！'
        }
    )
    submit = SubmitField('提交')