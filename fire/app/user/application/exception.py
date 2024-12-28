from core.exceptions.base import CustomException

class SignUpValidationException(CustomException):
    code = 422
    error_code = "SIGNUP_VALIDATION_ERR"

class EmailValidationException(SignUpValidationException):
    message = "email validation error"

class PasswordValidationException(SignUpValidationException):
    error_code = "PASSWORD_VALIDATION_ERR"

class PasswordNotEqualsException(SignUpValidationException):
    message = "password1 != password2"

class PasswordLengthUnsatisfiedException(SignUpValidationException):
    message = "password should be >= 10"

class AlreadySignedUpException(SignUpValidationException):
    message = "already signed up client"


class LoginValidationException(CustomException):
    code = 422
    error_code = "LOGIN_VALIDATION_ERR"

class NotFoundUserException(LoginValidationException):
    message = "User Not Found"

class PasswordWrongException(LoginValidationException):
    message = "Password Wrong"