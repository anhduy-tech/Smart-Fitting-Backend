from django.contrib import admin
from django.apps import apps

app = apps.get_app_config('app')

class GenericAdmin(admin.ModelAdmin):
    list_display = ['id', '__str__']
    search_fields = []
    list_filter = []
    readonly_fields = ()   # Để trống ban đầu

    def __init__(self, model, admin_site):
        super().__init__(model, admin_site)
        
        # Tự động lấy fields
        all_field_names = [f.name for f in model._meta.fields]
        
        # list_display
        self.list_display = ['id'] + [f for f in all_field_names if f != 'id'][:6]
        
        # search_fields
        self.search_fields = [f for f in all_field_names 
                            if f in ('name', 'title', 'email', 'username', 'phone')]
        
        # readonly_fields - chỉ lấy field thật sự tồn tại
        possible_readonly = {'created_at', 'updated_at', 'date_created', 'date_updated', 'created', 'modified'}
        self.readonly_fields = tuple(f for f in possible_readonly if f in all_field_names)

# Auto-register
for model_name, model in list(app.models.items()):
    admin.site.register(model, GenericAdmin)