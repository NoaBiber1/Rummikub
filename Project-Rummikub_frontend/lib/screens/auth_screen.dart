import 'package:firebase_ui_auth/firebase_ui_auth.dart';
import 'package:flutter/material.dart';

import '../widgets/auth_brand.dart';

/// Authentication screen using Firebase UI Auth email/password.
///
/// Visual chrome is custom; sign-in, registration, and password reset still
/// go through [EmailForm] and [ForgotPasswordView].
class AuthScreen extends StatefulWidget {
  const AuthScreen({super.key});

  @override
  State<AuthScreen> createState() => _AuthScreenState();
}

class _AuthScreenState extends State<AuthScreen> {
  AuthAction _action = AuthAction.signIn;
  final EmailAuthProvider _emailProvider = EmailAuthProvider();

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: authTheme(),
      child: FirebaseUIActions(
        actions: [
          ForgotPasswordAction((context, email) {
            Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (_) => _ForgotPasswordPage(email: email),
              ),
            );
          }),
        ],
        child: Scaffold(
          body: AuthBackdrop(
            child: SafeArea(
              child: Center(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.fromLTRB(24, 28, 24, 32),
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 420),
                    child: Column(
                      children: [
                        const RummikubHelperWordmark(),
                        const SizedBox(height: 20),
                        AuthFormCard(
                          child: Column(
                            children: [
                              _AuthModeToggle(
                                action: _action,
                                onChanged: (action) {
                                  setState(() => _action = action);
                                },
                              ),
                              const SizedBox(height: 20),
                              EmailForm(
                                action: _action,
                                provider: _emailProvider,
                                showPasswordVisibilityToggle: true,
                                style: const EmailFormStyle(
                                  signInButtonVariant: ButtonVariant.filled,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _AuthModeToggle extends StatelessWidget {
  const _AuthModeToggle({required this.action, required this.onChanged});

  final AuthAction action;
  final ValueChanged<AuthAction> onChanged;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0x33FFFFFF),
        borderRadius: BorderRadius.circular(22),
      ),
      child: Padding(
        padding: const EdgeInsets.all(4),
        child: Row(
          children: [
            Expanded(
              child: _ToggleTab(
                label: 'Login',
                selected: action == AuthAction.signIn,
                onTap: () => onChanged(AuthAction.signIn),
              ),
            ),
            Expanded(
              child: _ToggleTab(
                label: 'Sign Up',
                selected: action == AuthAction.signUp,
                onTap: () => onChanged(AuthAction.signUp),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ToggleTab extends StatelessWidget {
  const _ToggleTab({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: selected ? AuthColors.red : Colors.transparent,
      borderRadius: BorderRadius.circular(18),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(18),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Text(
            label,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontWeight: FontWeight.w700,
              color: selected ? Colors.white : const Color(0xE6FFFFFF),
            ),
          ),
        ),
      ),
    );
  }
}

class _ForgotPasswordPage extends StatelessWidget {
  const _ForgotPasswordPage({this.email});

  final String? email;

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: authTheme(),
      child: Scaffold(
        body: AuthBackdrop(
          child: SafeArea(
            child: Column(
              children: [
                Align(
                  alignment: Alignment.centerLeft,
                  child: IconButton(
                    onPressed: () => Navigator.of(context).pop(),
                    icon: const Icon(Icons.arrow_back_rounded),
                    color: Colors.white,
                  ),
                ),
                Expanded(
                  child: Center(
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.fromLTRB(24, 8, 24, 32),
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 420),
                        child: Column(
                          children: [
                            const RummikubHelperWordmark(compact: true),
                            const SizedBox(height: 24),
                            AuthFormCard(
                              child: ForgotPasswordView(email: email),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
