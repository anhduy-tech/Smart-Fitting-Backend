from django.urls import re_path
from app import views

urlpatterns = [
    # --- System & Data Tools ---
    re_path('get-model/$', views.get_model),
    re_path('data/(?P<name>.+)/$', views.data_list),
    re_path('data-detail/(?P<name>.+)/(?P<pk>[0-9]+)/$', views.data_detail),
    re_path('import-data/(?P<name>.+)/$', views.import_data),
    re_path('delete-data/(?P<name>.+)/$', views.delete_data),
    re_path('upload/$', views.upload),
    re_path('download/$', views.download),

    # --- Auth & Profile ---
    re_path('auth/register/$', views.register),
    re_path('auth/login/$', views.login),
    re_path('auth/send-otp/$', views.send_otp),
    re_path('auth/verify-otp/$', views.verify_otp),
    re_path('auth/forgot-password/$', views.forgot_password),
    re_path('auth/change-password/$', views.change_password),
    re_path('auth/profile/$', views.profile),
    re_path('auth/update-profile/$', views.update_profile),
    re_path('user/measurements/$', views.update_measurements),

    # --- Portrait & AI Try-on ---
    re_path('portrait/upload/$', views.upload_portrait),
    re_path('portrait/list/$', views.portrait_list),
    re_path('portrait/remove-background/(?P<pk>[0-9]+)/$', views.portrait_remove_background),
    re_path('portrait/delete/(?P<pk>[0-9]+)/$', views.portrait_delete),
    re_path('tryon/generate/$', views.generate_tryon),
    re_path('tryon/status/(?P<pk>[0-9]+)/$', views.tryon_status),
    re_path('tryon/history/$', views.tryon_history),
    re_path('tryon/delete/(?P<pk>[0-9]+)/$', views.tryon_delete),

    # --- Products & Frames ---
    re_path('products/$', views.product_list),
    re_path('products/(?P<pk>[0-9]+)/$', views.product_detail),
    re_path('products/search/$', views.product_search),
    re_path('categories/$', views.category_list),
    re_path('frames/$', views.frame_list),
    re_path('frame-categories/$', views.frame_category_list),

    # --- Cart & Orders ---
    re_path('cart/$', views.cart_list),
    re_path('cart/add/$', views.cart_add),
    re_path('cart/update/(?P<pk>[0-9]+)/$', views.cart_update),
    re_path('cart/delete/(?P<pk>[0-9]+)/$', views.cart_delete),
    re_path('favourite/$', views.favourite_list),
    re_path('favourite/toggle/$', views.favourite_toggle),
    re_path('order/create/$', views.order_create),
    re_path('order/list/$', views.order_list),
    re_path('order/detail/(?P<pk>[0-9]+)/$', views.order_detail),
    re_path('order/cancel/(?P<pk>[0-9]+)/$', views.order_cancel),
    re_path('payment/qr/(?P<pk>[0-9]+)/$', views.payment_qr),
    re_path('payment/callback/$', views.payment_callback),

    # --- Support & Notifications ---
    re_path('support/create/$', views.support_create),
    re_path('support/list/$', views.support_list),
    re_path('support/detail/(?P<pk>[0-9]+)/$', views.support_detail),
    re_path('support/reply/(?P<pk>[0-9]+)/$', views.support_reply),
    re_path('notifications/$', views.notification_list),
    re_path('notifications/read/(?P<pk>[0-9]+)/$', views.notification_read),
    re_path('notifications/read-all/$', views.notification_read_all),
    re_path('slides/$', views.slide_list),

    # --- Admin Panel ---
    re_path('admin/product/create/$', views.admin_product_create),
    re_path('admin/product/update/(?P<pk>[0-9]+)/$', views.admin_product_update),
    re_path('admin/product/delete/(?P<pk>[0-9]+)/$', views.admin_product_delete),
    re_path('admin/category/create/$', views.admin_category_create),
    re_path('admin/frame/create/$', views.admin_frame_create),
    re_path('admin/frame-category/create/$', views.admin_frame_category_create),
    re_path('admin/orders/$', views.admin_order_list),
    re_path('admin/order/update-status/(?P<pk>[0-9]+)/$', views.admin_order_update_status),
    re_path('admin/users/$', views.admin_user_list),
    re_path('admin/support/$', views.admin_support_list),
    re_path('admin/support/reply/(?P<pk>[0-9]+)/$', views.admin_support_reply),
    re_path('admin/background-remove/$', views.admin_background_remove),
    re_path('admin/settings/$', views.admin_settings),
    re_path('admin/dashboard/$', views.admin_dashboard),
]