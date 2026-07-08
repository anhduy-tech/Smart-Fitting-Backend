import os, ast, re, mimetypes
import cv2, requests, secrets, csv
from django.apps import apps
from rest_framework import status, serializers
from rest_framework.decorators import api_view
from rest_framework.response import Response
from app.models import *
from django.contrib.auth.hashers import make_password, check_password
from django.http import FileResponse
from django.db import models as aggregator
from django.db.models.functions import Concat, RowNumber, Coalesce, Length, ExtractDay, TruncDate, Cast
from django.db.models import F, Value, TextField, CharField, Q, Count, Min, Max, Sum, Func, JSONField, FloatField, Case, When, OuterRef, Subquery, NOT_PROVIDED
from datetime import datetime, date, timedelta
from django.contrib.postgres.aggregates import ArrayAgg
from django.db.models.expressions import Window
from operator import add, sub, mul, truediv
from django.http.response import StreamingHttpResponse, HttpResponse
from wsgiref.util import FileWrapper
from django.core.cache import cache
from django.views.decorators.csrf import csrf_exempt
from django.http.request import QueryDict
from django.http import JsonResponse
from django.core.cache import cache
import json
import io
from django.apps import apps
from django.db.models import CharField, TextField

from rest_framework.response import Response
from rest_framework import status

def querydict_to_nested_dict(query_dict):
    """
    Thay the cho app/querydict.py (khong ton tai trong project nay).
    Chi convert QueryDict -> dict phang (cac endpoint dang dung ham nay
    - data_detail, import_data - chi doc key phang nhu 'values', 'summary',
    'sort'..., khong can nested), lay gia tri dau tien cho moi key.
    """
    if not query_dict:
        return {}
    return {key: query_dict.get(key) for key in query_dict.keys()}


limit_rows = 2000
perpage = 20
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_folder = os.path.join(BASE_DIR, "static")
TextField.register_lookup(Length, 'length')
CharField.register_lookup(Length, 'length')
BLOCKED_HOST = ["api.utopia.com.vn", "dev.api.utopia.com.vn"] 

#=============================================================================
def check_access(request):
    # origin = request.headers.get("Origin")
    # host = request.get_host()
    # user_agent = request.META.get("HTTP_USER_AGENT", "").lower()
    # if "dart" in user_agent:
    #     return True

    # if not origin and host in BLOCKED_HOST:
    #     return False
    
    return True
    

# Get limit rows
#=============================================================================
def get_limit_rows(rows, page, onpage):
    total_rows = rows.count()
    full_data = True if (total_rows <= limit_rows) or page == -1 else False

    if full_data == True and onpage != None:
        full_data = False if total_rows > onpage else full_data

    if full_data == False:
        onpage = onpage if onpage != None else perpage
        rows = rows[(page-1) * onpage : page * onpage]
    return total_rows, full_data, rows

#=============================================================================
#--- get ip ---
def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

#=============================================================================
def get_serializer(name):
    try:
        Model = apps.get_model('app', name)
    except:
        return None, None

    class GenericSerializer(serializers.ModelSerializer):
        class Meta:
            model = Model
            fields = '__all__'

        def create(self, validated_data):
            return Model.objects.create(**validated_data)
        
        def update(self, instance, validated_data):
            for attr, value in validated_data.items():
                setattr(instance, attr, value)

            instance.save()
            return instance
    return Model, GenericSerializer

#=============================================================================
def update_increment(name):
    value = cache.get(name)
    if value == None:
        cache.set(name, {'date': date.today(), 'current': 1}, timeout=2592000)
    else:
        if value['date'] == date.today():
            value['current'] += 1
            cache.set(name, value, timeout=2592000)
        else:
            cache.set(name, {'date': date.today(), 'current': 1}, timeout=2592000)

#=============================================================================
def current_increment(name, prefix):
    value = cache.get(name)
    if value == None:
        cache.set(name, {'date': date.today(), 'current': 0}, timeout=2592000)
        next = 1
        current_date = date.today()
    else:
        next = value['current'] + 1
        current_date = value['date']
    dt = datetime.strptime(str(current_date), '%Y-%m-%d')
    formatted = dt.strftime('%d%m%y')
    return f"{prefix}{formatted}{next:03d}"

#=============================================================================
@api_view(['GET'])
def get_increment(request, name, prefix):
    value = cache.get(name)
    if value == None:
        cache.set(name, {'date': date.today(), 'current': 0}, timeout=2592000)
        next = 1
        current_date = date.today()
    else:
        next = value['current'] + 1
        current_date = value['date']
    dt = datetime.strptime(str(current_date), '%Y-%m-%d')
    formatted = dt.strftime('%d%m%y')
    return Response(f"{prefix}{formatted}{next:03d}")

#=============================================================================
@api_view(['GET'])
def get_increment_next(request, name, prefix):
    value = cache.get(name)
    if value == None:
        cache.set(name, {'date': date.today(), 'current': 0}, timeout=2592000)
        next = 1
        current_date = date.today()
    else:
        next = value['current'] + 1
        current_date = value['date']
        cache.set(name, {'date': date.today(), 'current': next}, timeout=2592000)
    dt = datetime.strptime(str(current_date), '%Y-%m-%d')
    formatted = dt.strftime('%d%m%y')
    return Response(f"{prefix}{formatted}{next:03d}")

#=============================================================================
def subquery(value):
    Model, serializer_class = get_serializer(value['subquery']['model'])
    column = value['subquery']['column']
    field = value['field']
    func = value['type']
    filter = value['filter'] if 'filter' in value else {}
    filter[column] = OuterRef('pk')
    query = Model.objects.filter(**filter
    ).annotate(
        value_query=Func(F(field), function=func)
    ).values('value_query')
    return query

#=============================================================================
def base_query(rows, values, summary, distinct_values):
    need_serializer = True
    funcs = {'Count': Count, 'Min': Min, 'Max': Max, 'Sum': Sum, 'ExtractDay': ExtractDay}
    if  values != None:
        rows = rows.values(*values)
        need_serializer = False
    if summary =='distinct':
        distinct_values = '' if distinct_values==None else distinct_values.split(',')
        rows = rows.distinct(*distinct_values)
    elif summary == 'count':
        rows = rows.count()
    elif summary =='annotate':
        ele = {}
        for key, value in ast.literal_eval(distinct_values).items():
            if isinstance(value, str) == True:
                reducer = getattr(aggregator, value)
                ele[key] = reducer(key)
            else:
                if value['type'] == 'Concat':
                    arr = []
                    char = value['char'] if 'char' in value else ' / '
                    for idx, field in enumerate(value['field']):
                        arr.append(F(field))
                        arr.append(Value(char)) if idx < len(value['field']) - 1 else False
                    reducer = Concat(*arr, output_field=CharField())
                if value['type'] == 'RowNumber':
                    reducer = Window(expression=RowNumber())
                elif value['type'] == 'ArrayAgg':
                    arr1 = []
                    for field in value['field']:
                        arr1.append(Value(field))
                        arr1.append(field)
                    reducer = ArrayAgg(Func(*arr1, function="jsonb_build_object", output_field=JSONField()), distinct=True)
                elif value['type'] in funcs:
                    func = funcs[value['type']]
                    arr = Q()
                    if 'subquery' in value:
                        ele[key] = Subquery(subquery(value))
                        continue;
                    if 'filter' in value:
                        for fkey in value['filter']:
                            arr.add(Q(**{fkey: value['filter'][fkey]}), Q.AND)
                    if 'formula' in value:
                        reducer = None; exp = None
                        operator = {'+': add, '-': sub, '*': mul, ':': truediv, 'or': Coalesce}
                        keyword = {'now': Value(datetime.now())}
                        for field in value['formula']:
                            if field in operator:
                                exp = operator[field]
                            else:
                                if isinstance(field, str):
                                    expression = keyword[field] if field in keyword else F(field)
                                else:
                                    expression = Value(field)
                                if value['type'] == 'ExtractDay':
                                    expression = TruncDate(expression)
                                reducer = expression if reducer==None else exp(reducer, expression)
                        reducer = func(reducer, filter=arr, output_field=FloatField())
                    else:
                        reducer = func(value['field'], filter=arr, distinct= True if 'distinct' in value else False)
                # reducer
                ele[key] = reducer
        # query
        rows = rows.annotate(**ele)
    elif summary == 'aggregate':
        ele = {}
        for key, value in ast.literal_eval(distinct_values).items():
            arr = Q()
            if 'filter' in value:
                for fkey in value['filter']:
                    arr.add(Q(**{fkey: value['filter'][fkey]}), Q.AND)
            func = funcs[value['type']]
            reducer = func(value['field'], filter=arr)
            ele[key] = reducer
        rows = rows.aggregate(**ele)

    # return
    return rows, need_serializer

#=============================================================================
def calculate(rows, calculation):
    divcheck = None
    ele = {}
    for key, value in ast.literal_eval(calculation).items():
        reducer = None; exp = None
        operator = {'+': add, '-': sub, '*': mul, ':': truediv, 'or': Coalesce}
        for field in value['formula']:
            if field in operator:
                exp = operator[field]
            else: 
                expression = F(field) if isinstance(field, str) else Value(field)
                reducer = expression if reducer==None else exp(reducer, expression)
                if exp == truediv:
                    divcheck = {field: 0}
        if divcheck:
            reducer = Case(When(**divcheck, then=Value(None)), **{'default': reducer}, output_field=FloatField())
        ele[key] = reducer
    rows = rows.annotate(**ele)
    return rows

#=============================================================================
def final_result(rows, calculation=None, final_filter=None, final_exclude=None, sort=None):
    if calculation:
        rows = calculate(rows, calculation)

    if final_filter:
        filter_list = Q()
        for key, value in final_filter.items():
            if isinstance(value, dict) and value.get('type') == 'F':
                filter_list.add(Q(**{key: F(value['field'])}), Q.AND)
            else:
                filter_list.add(Q(**{key: value}), Q.AND)
        rows = rows.filter(filter_list)
        
    if final_exclude:
        exclude_list = Q()
        for key, value in final_exclude.items():
            if isinstance(value, dict) and value.get('type') == 'F':
                exclude_list.add(Q(**{key: F(value['field'])}), Q.AND)
            else:
                exclude_list.add(Q(**{key: value}), Q.AND)
        rows = rows.exclude(exclude_list)
    # sort
    if sort:
        rows = rows.order_by(*sort)
    return rows

#=============================================================================
@api_view(['GET', 'POST'])
def data_list(request, name):
    Model, serializer_class = get_serializer(name)
    if Model == None:
        return Response(status=status.HTTP_400_BAD_REQUEST)
    
    # check access
    if check_access(request)==False:
        return JsonResponse({"detail": "Direct access not allowed"}, status=403)

    filter = request.query_params['filter'] if request.query_params.get('filter') != None else None
    values = request.query_params['values'] if request.query_params.get('values') != None else None
    values = values if values==None else values.split(',')
    summary = request.query_params['summary'] if request.query_params.get('summary') != None else None
    page = int(request.query_params['page']) if request.query_params.get('page') != None else 1
    onpage = int(request.query_params['perpage']) if request.query_params.get('perpage') != None else None
    sort = request.query_params['sort'] if request.query_params.get('sort') != None else None
    sort = None if sort==None else sort.split(',')
    distinct_values = request.query_params['distinct_values'] if request.query_params.get('distinct_values') != None else None
    filter_or = request.query_params['filter_or'] if request.query_params.get('filter_or') != None else None
    exclude = request.query_params['exclude'] if request.query_params.get('exclude') != None else None
    calculation = request.query_params['calculation'] if request.query_params.get('calculation') != None else None
    final_filter = request.query_params['final_filter'] if request.query_params.get('final_filter') != None else None
    final_exclude = request.query_params['final_exclude'] if request.query_params.get('final_exclude') != None else None
    cache_info = request.query_params['cache'] if request.query_params.get('cache') != None else None

    if cache_info != None:
        cache_info = ast.literal_eval(cache_info)
        cache_value = cache.get(cache_info["key"])
        if cache_value != None:
            return Response({'total_rows': len(cache_value), 'full_data': True, 'rows': cache_value})

    need_serializer = True    
    filter_list = Q()
    if filter_or != None:
        field_map = {f.name: f for f in Model._meta.get_fields()}
        for key, value in ast.literal_eval(filter_or).items():
            lookup_parts = key.split('__')
            base_field_name = lookup_parts[0]

            if base_field_name not in field_map:
                continue
            
            field_obj = field_map[base_field_name]

            if field_obj.is_relation and len(lookup_parts) == 2 and lookup_parts[1] == 'icontains':
                continue

            is_numeric = isinstance(field_obj, (aggregator.IntegerField, aggregator.DecimalField, aggregator.FloatField))
            is_date = isinstance(field_obj, (aggregator.DateField, aggregator.DateTimeField))
            if (is_numeric or is_date) and key.endswith('__icontains'):
                continue
            
            filter_list.add(Q(**{key: value}), Q.OR)

    if filter != None:
        for key, value in ast.literal_eval(filter).items():
            if isinstance(value, dict) == True:
                if value['type'] == 'F':
                    filter_list.add(Q(**{key: F(value['field'])}), Q.AND)
            else:
                filter_list.add(Q(**{key: value}), Q.AND)
                
    if request.method == 'GET':
        rows = Model.objects.all() if len(filter_list) == 0 else Model.objects.filter(filter_list)
        if exclude != None:
            exclude_list = Q()
            for key, value in ast.literal_eval(exclude).items():
                if isinstance(value, dict) == True:
                    if value['type'] == 'F':
                        exclude_list.add(Q(**{key: F(value['field'])}), Q.AND)
                else:
                    exclude_list.add(Q(**{key: value}), Q.AND)
            rows = rows.exclude(exclude_list)        
        rows, need_serializer = base_query(rows, values, summary, distinct_values)
        rows = final_result(rows, calculation, final_filter, final_exclude, sort)
        if summary == 'count' or summary == 'aggregate':
            return Response({'total_rows': 1, 'full_data': True, 'rows': rows})
        total_rows, full_data, rows = get_limit_rows(rows, page, onpage)
        if need_serializer == True:
            rows = serializer_class(rows, many=True).data
            if cache_info:
                # Lưu giá trị vào cache
                cache.set(cache_info['key'], rows, timeout=cache_info['timeout'])
        return Response({'total_rows': total_rows, 'full_data': full_data, 'rows': rows})

    elif request.method == 'POST':
        serializer = serializer_class(data = request.data)
        if serializer.is_valid():
            serializer.save()
            data = serializer.data
            # update increment
            update_increment(name)
            if values != None:
                rows = Model.objects.filter(id=data['id'])
                rows, need_serializer = base_query(rows, values, summary, distinct_values)
                rows = final_result(rows, calculation, final_filter, final_exclude, sort)
                if need_serializer == True:
                    rows = serializer_class(rows, many=True).data
                return Response(rows[0])
            return Response(data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#=============================================================================
@api_view(['GET', 'PUT', 'DELETE', 'PATCH'])
def data_detail(request, name, pk):
    # check access
    if check_access(request) == False:
        return JsonResponse({"detail": "Direct access not allowed"}, status=403)
    
    query_params = querydict_to_nested_dict(request.query_params)
    query_params = {} if not query_params else query_params
    values = query_params['values'] if query_params.get('values') != None else None
    values = values if values == None else values.split(',')
    summary = query_params['summary'] if query_params.get('summary') != None else None
    sort = query_params['sort'] if query_params.get('sort') != None else None
    sort = sort if sort == None else sort.split(',')
    distinct_values = query_params['distinct_values'] if query_params.get('distinct_values') != None else None
    calculation = query_params['calculation'] if query_params.get('calculation') != None else None
    
    Model, serializer_class = get_serializer(name)
    if Model == None:
        return Response(status=status.HTTP_400_BAD_REQUEST)
    
    try:
        obj = Model.objects.get(pk=pk)
    except Model.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        serializer = serializer_class(obj)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = serializer_class(obj, data=request.data)
        if serializer.is_valid():
            serializer.save()
            data = serializer.data
            if values != None:
                rows = Model.objects.filter(id=data['id'])
                rows, need_serializer = base_query(rows, values, summary, str(distinct_values))
                rows = final_result(rows, calculation)
                return Response(rows.first())
            else:
                return Response(data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'PATCH':
        serializer = serializer_class(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            data = serializer.data
            if values != None:
                rows = Model.objects.filter(id=data['id'])
                rows, need_serializer = base_query(rows, values, summary, str(distinct_values))
                rows = final_result(rows, calculation)
                return Response(rows.first())
            else:
                return Response(data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        if name == 'File' or name == 'Image' or name == 'Video':
            file_name = static_folder + ('/' + name.lower() + 's/') + obj.file
            if os.path.exists(file_name):
                os.remove(file_name)
        try:
            obj.delete()
        except Exception as e:
            print(e)
            return Response(data=str(e), status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_204_NO_CONTENT)

#====================================================================
@api_view(['POST'])
def import_data(request, name):
    Model, serializer_class = get_serializer(name)
    if Model == None:
        return Response(status=status.HTTP_400_BAD_REQUEST)
    
    # check access
    if check_access(request)==False:
        return JsonResponse({"detail": "Direct access not allowed"}, status=403)
     
    data = [request.data.dict()] if type(request.data) == QueryDict else request.data
    query_params = querydict_to_nested_dict(request.query_params)
    query_params = {} if not query_params else query_params
    values = query_params['values'] if query_params.get('values') != None else None
    values = values if values==None else values.split(',')
    summary = query_params['summary'] if query_params.get('summary') != None else None
    sort = query_params['sort'] if query_params.get('sort') != None else None
    sort = sort if sort==None else sort.split(',')
    distinct_values = query_params['distinct_values'] if query_params.get('distinct_values') != None else None
    calculation = query_params['calculation'] if query_params.get('calculation') != None else None

    error = False
    return_data = []
    for row in data:
        try:
            if 'id' in row:
                ele = Model.objects.filter(pk=row['id']).first()
                if ele == None:
                    serializer = serializer_class(data = row, partial=True) # insert
                else:
                    serializer = serializer_class(ele, data=row, partial=True) # update
            else:
                serializer = serializer_class(data = row)
            if serializer.is_valid():
                serializer.save()
                return_data.append(serializer.data if values == None else serializer.data['id'])
            else:
                row['error'] = True
                row['note'] = serializer.errors
                error = True
        except:
            row['error'] = True
            error = True

    if error == True:
        return Response(data)
    elif values == None:
        return Response(return_data)
    else:
        rows = Model.objects.filter(id__in = return_data)
        rows, need_serializer = base_query(rows, values, summary, str(distinct_values))
        rows = final_result(rows, calculation)
        return Response(rows)

#=============================================================================
@api_view(['POST'])
def delete_data(request, name):
    Model, serializer_class = get_serializer(name)
    if Model == None:
        return Response(status=status.HTTP_400_BAD_REQUEST)
 
    from django.http.request import QueryDict
    data = [request.data.dict()] if type(request.data) == QueryDict else request.data
    for row in data:
        try:
            if 'id' in row:
                ele = Model.objects.filter(pk=row['id']).first()
                if ele == None:
                    row['error'] = True
                    row['note'] = 'id=' + str(row['id']) +  ' not exist'
                else:
                    ele.delete()
                    row['deleted'] = True
            else:
                row['error'] = True
                row['note'] = 'field id not found'
        except Exception as e:
            row['error'] = True
            row['note'] = str(e)
    # return
    return Response(data)

#=============================================================================
@api_view(['POST'])
def get_hash(request):
    if request.method == 'POST':
        text = request.data['text']
        password = make_password(text)
        return Response({'total_rows': 1, 'full_data': True , 'rows': [password]})
    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
@api_view(['GET'])
def login(request):
    if request.method == 'GET':
        filter = request.query_params['filter'] if request.query_params.get('filter') != None else None
        filter = ast.literal_eval(filter)
        values = request.query_params['values'] if request.query_params.get('values') != None else None
        values = values if values==None else values.split(',')
        need_serializer = False

        if values == None:
            user = User.objects.filter(username=filter['username']).first()
            need_serializer = True
        else:
            user = User.objects.filter(username=filter['username']).values(*values).first()

        if user == None:
            return Response(None)

        result = check_password(filter['password'], user.password if need_serializer == True else user['password'])
        if result == False:
            return Response(None)
        
        if need_serializer == True:
            Model, serializer_class = get_serializer('User')
            serializer = serializer_class(user)
            return Response({'total_rows': 1, 'full_data': True , 'rows': serializer.data})
        return Response({'total_rows': 1, 'full_data': True , 'rows': user})
    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
@api_view(['POST'])
def signin(request):
    username = request.data['username']
    password = request.data['password']
    user = User.objects.filter(username=username).first()
    if user:
        result = check_password(password, user.password)
        if result == False:
            return Response("invalid")
        else:
            info = User.objects.filter(pk=user.id).values('id','username','avatar','fullname','auth_status','auth_status__code','auth_status__name').first()
            return Response(info)
    # invalid    
    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
@api_view(['POST'])
def check_pin(request):
    username = request.data['username']
    pin = request.data['pin']
    user = User.objects.filter(username=username).first()
    if user:
        result = check_password(pin, user.pin)
        if result == False:
            return Response("invalid")
        else:
            info = User.objects.filter(pk=user.id).values('id','username','avatar','fullname','auth_status','auth_status__code','auth_status__name').first()
            return Response(info)
    # invalid    
    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
def convert_webp(name, convert=80):
    jpg_file_path = '{}/files/{}'.format(static_folder, name)
    arr = name.split('.')
    text = '.' + arr[len(arr)-1]
    if text == '.webp':
        return name
    new_name = name.replace(text, '.webp')
    webp_file_path = '{}/files/{}'.format(static_folder, new_name)
    # Read the JPG image
    jpg_img = cv2.imread(jpg_file_path)
    # Save the image in JPG format with specific quality
    cv2.imwrite(webp_file_path, jpg_img, [int(cv2.IMWRITE_WEBP_QUALITY), int(convert)])
    os.remove(jpg_file_path)
    return new_name

#=============================================================================
@api_view(['POST'])
def upload(request):
    upload_folder = static_folder + '/files/'
    Model, serializer_class = get_serializer('File')
    # start
    if request.method == 'POST':
        file = request.data['file']
        filename = request.data['filename']
        convert = request.data['convert'] if 'convert' in request.data else None
        doc_type = request.data['doc_type'] if 'doc_type' in request.data else None

        # check type
        type = 1
        if request.data['type'] == 'video':
            type = 3
        elif request.data['type'] == 'image':
           type = 2
        # start upload
        try:
            with open(upload_folder + filename, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)     
            if type == 2 and convert != '0':
                filename = convert_webp(filename, convert)
            # save record
            data = {'type': type, 'user': request.data['user'], 'name': request.data['name'], 'file': filename, 
                    'size': request.data['size'], "doc_type": doc_type}
            serializer = serializer_class(data = data)
            if serializer.is_valid():
                serializer.save()
            else:
                print(serializer.errors)
            return Response({'total_rows': 1, 'full_data': True, 'rows': [serializer.data]})
        except:
            Response(status = status.HTTP_400_BAD_REQUEST)
    # return
    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
@csrf_exempt
@api_view(['GET'])
def download(request):
    upload_folder = static_folder + '/files/'
    name = request.query_params['name'] if request.query_params.get('name') != None else None
    type = request.query_params['type'] if request.query_params.get('type') != None else None
    if type=='contract':
        upload_folder = static_folder + '/contract/'

    if  name == None:
        return Response(status = status.HTTP_400_BAD_REQUEST)

    if request.method == 'GET':
        if os.path.exists(upload_folder + name):
            response = FileResponse(open(upload_folder + name, 'rb'), as_attachment=True)
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Expose-Headers"] = "Content-Disposition"
            return response

    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
@csrf_exempt
@api_view(['GET'])
def download_contract(request, name):
    upload_folder = static_folder + '/contract/'
    if  name == None:
        return Response(status = status.HTTP_400_BAD_REQUEST)

    if request.method == 'GET':
        if os.path.exists(upload_folder + name):
            response = FileResponse(open(upload_folder + name, 'rb'), as_attachment=True)
            response["Access-Control-Allow-Origin"] = "*"
            response["Access-Control-Expose-Headers"] = "Content-Disposition"
            return response

    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
@api_view(['POST'])
def batch_upload(request):
    folder = request.data['folder']
    file = request.data['file']
    filename = request.data['name']
    upload_folder = static_folder + '/' + folder + '/'
    try:
        with open(upload_folder + filename, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)
        return Response({'filename': filename, 'path': upload_folder + '/' + filename})
    except:
        Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
def write_log(data):
    Model, serializer_class = get_serializer('Log')
    serializer = serializer_class(data=data)
    if serializer.is_valid():
        serializer.save()
    else:
        print(serializer.errors)

#=============================================================================
@api_view(['POST'])
def auth_token(request):
    data = request.data
    Model, serializer_class = get_serializer('Token')
    row = Model.objects.filter(token=data['token']).first()
    if row:
        serializer = serializer_class(row)
        return Response(serializer.data)
    # new
    data['ip'] = get_client_ip(request)
    # get ip info
    url = "https://ipinfo.io/{}?token=1cc0a688798cf7".format(data['ip'])
    try:
        rs = requests.get(url, timeout=2)
        obj = rs.json()
        for key in obj:
            data[key] = obj[key]
    except:
        print("An exception occurred")
    # save
    print(data)
    serializer = serializer_class(data = data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    else:
        return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
class RangeFileWrapper(object):
    def __init__(self, filelike, blksize=8192, offset=0, length=None):
        self.filelike = filelike
        self.filelike.seek(offset, os.SEEK_SET)
        self.remaining = length
        self.blksize = blksize

    def close(self):
        if hasattr(self.filelike, 'close'):
            self.filelike.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.remaining is None:
            # If remaining is None, we're reading the entire file.
            data = self.filelike.read(self.blksize)
            if data:
                return data
            raise StopIteration()
        else:
            if self.remaining <= 0:
                raise StopIteration()
            data = self.filelike.read(min(self.remaining, self.blksize))
            if not data:
                raise StopIteration()
            self.remaining -= len(data)
            return data

#=============================================================================
def stream_video(request, path):
    range_re = re.compile(r'bytes\s*=\s*(\d+)\s*-\s*(\d*)', re.I)
    range_header = request.META.get('HTTP_RANGE', '').strip()
    range_match = range_re.match(range_header)
    path = static_folder + '/files/' + path
    size = os.path.getsize(path)
    content_type, encoding = mimetypes.guess_type(path)
    content_type = content_type or 'application/octet-stream'
    if range_match:
        first_byte, last_byte = range_match.groups()
        first_byte = int(first_byte) if first_byte else 0
        last_byte = int(last_byte) if last_byte else size - 1
        if last_byte >= size:
            last_byte = size - 1
        length = last_byte - first_byte + 1
        resp = StreamingHttpResponse(RangeFileWrapper(open(path, 'rb'), offset=first_byte, length=length), status=206, content_type=content_type)
        resp['Content-Length'] = str(length)
        resp['Content-Range'] = 'bytes %s-%s/%s' % (first_byte, last_byte, size)
    else:
        resp = StreamingHttpResponse(FileWrapper(open(path, 'rb')), content_type=content_type)
        resp['Content-Length'] = str(size)
    resp['Accept-Ranges'] = 'bytes'
    return resp

#=============================================================================
@api_view(['GET'])
def get_cache(request, name):
    value = cache.get(name)
    return Response(value)

#=============================================================================
@api_view(['GET'])
def delete_cache(request, name):
    cache.delete(name)
    return Response(status=status.HTTP_200_OK)

#=============================================================================
@api_view(['GET'])
def get_model(request):
    # Lấy toàn bộ models trong project
    arr = []
    all_models = apps.get_models()
    for model in all_models:
        if str(model._meta).find("auth.")>=0:
            continue
        
        # next
        fields = []
        for field in model._meta.get_fields():        
            info = {'name': field.name, 'type': field.get_internal_type()}
            info['unique'] = field.unique if hasattr(field, 'unique') else None
            info['null'] = field.null if hasattr(field, 'null') else None
            if hasattr(field, 'default'):
                info['default'] = str(field.default) if field.default is not NOT_PROVIDED else None

            # foreign keys
            if field.get_internal_type() == 'ForeignKey':
                model_class = field.remote_field.model
                info['model'] = model_class.__name__
                if isinstance(field, (models.OneToOneRel, models.ManyToOneRel)):
                    info['relation'] = type(field).__name__
                else:
                    info['relation'] = "OneToManyRel"

            # insert
            fields.append(info)
        arr.append({'model': model.__name__, 'fields':fields})
    return Response({'total_rows': len(arr), 'full_data': True, 'rows': arr})

#=============================================================================
@api_view(['GET'])
def get_password(request, text):
    password = make_password(text)
    return Response(password)

#=============================================================================
@api_view(['GET'])
def export_csv(request, name):
    filter = request.query_params['filter'] if request.query_params.get('filter') != None else None
    values = request.query_params['values'] if request.query_params.get('values') != None else None
    values = values if values==None else values.split(',')
    summary = request.query_params['summary'] if request.query_params.get('summary') != None else None
    sort = request.query_params['sort'] if request.query_params.get('sort') != None else None
    sort = None if sort==None else sort.split(',')
    distinct_values = request.query_params['distinct_values'] if request.query_params.get('distinct_values') != None else None
    filter_or = request.query_params['filter_or'] if request.query_params.get('filter_or') != None else None
    exclude = request.query_params['exclude'] if request.query_params.get('exclude') != None else None
    calculation = request.query_params['calculation'] if request.query_params.get('calculation') != None else None
    final_filter = request.query_params['final_filter'] if request.query_params.get('final_filter') != None else None
    final_exclude = request.query_params['final_exclude'] if request.query_params.get('final_exclude') != None else None
    fields = request.query_params['fields'] if request.query_params.get('fields') != None else None

    # get model
    Model, serializer_class = get_serializer(name)
    if Model == None:
        return Response(status=status.HTTP_400_BAD_REQUEST)

    # filter
    filter_list = Q()
    if filter_or != None:
        for key, value in ast.literal_eval(filter_or).items():
            filter_list.add(Q(**{key: value}), Q.OR)

    if filter != None:
        for key, value in ast.literal_eval(filter).items():
            if isinstance(value, dict) == True:
                if value['type'] == 'F':
                    filter_list.add(Q(**{key: F(value['field'])}), Q.AND)
            else:
                filter_list.add(Q(**{key: value}), Q.AND)
    
    rows = Model.objects.all() if len(filter_list) == 0 else Model.objects.filter(filter_list)
    if exclude != None:
        exclude_list = Q()
        for key, value in ast.literal_eval(exclude).items():
            if isinstance(value, dict) == True:
                if value['type'] == 'F':
                    exclude_list.add(Q(**{key: F(value['field'])}), Q.AND)
            else:
                exclude_list.add(Q(**{key: value}), Q.AND)
        rows = rows.exclude(exclude_list)        
    rows, need_serializer = base_query(rows, values, summary, distinct_values)
    rows = final_result(rows, calculation, final_filter, final_exclude, sort)
    columns = ast.literal_eval(fields)
    dtFields = {}
    for field in Model._meta.get_fields():
        if field.get_internal_type() == 'DateTimeField':
            dtFields[field.name] = 'DateTimeField'

    # Lấy tất cả dữ liệu từ model Customer và ghi vào CSV
    output = []
    for row in rows:
        arr = []
        for o in columns:
            name = o.get('name')
            if type(row) == dict:
                val = row[name] if name in row else None
            else:
                val = getattr(row, name)       
            if dtFields.get(name) != None and val != None:
                val = row[name].strftime("%d/%m/%Y %H:%M:%S")
            arr.append(val)
        output.append(arr)

    # Thiết lập HTTP response để tải file CSV
    response = HttpResponse(content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'

    # Thêm BOM cho UTF-8-SIG (tùy chọn, cần cho Excel)
    response.write(b'\xef\xbb\xbf')  # BOM cho UTF-8-SIG    
    writer = csv.writer(response, lineterminator='\n')
    writer.writerow([o.get('label') for o in columns])
    writer.writerows(output)
    return response

#=============================================================================
@api_view(['GET', 'POST'])
def get_otp(request):
    if request.method == 'GET':
        max = 10
        phone = request.query_params['phone'] if request.query_params.get('phone') != None else None
        if phone == None:
            return Response(status = status.HTTP_400_BAD_REQUEST)
        
        # check sent today
        rows = Phone_Otp.objects.filter(phone=phone, create_time__date=datetime.now().strftime("%Y-%m-%d"))
        if rows.count() > max:
            return Response({"status": "error", "text": "The OTP sending limit for today has been reached. Please wait until tomorrow to continue"})
          
        code = current_increment('Phone_Otp', 'OTP')
        otp = ''.join(str(secrets.randbelow(10)) for _ in range(6))
        setting = System_Setting.objects.filter(classify="template", code="otp").first()
        content = setting.vi
        content = content.replace("<otp>", otp)
        valid_to = datetime.now() + timedelta(minutes=5)
        data = {"code": code, "phone": phone, "otp": otp, "status": 1, "sms_content": content, "sms_fee": 1000, "valid_to": valid_to}
        obj = {"phone": phone, "message": content, "shop": 1, "type": 1, "agent": 3}
        url = "https://accountapi.loan247.vn/send-sms/"
        response = requests.post(url, obj)
        try:
            data['sms_info'] = response.json()
            data['result'] = 2
        except Exception as e:
            print(e)
            data['result'] = 3

        Model, serializer_class = get_serializer("Phone_Otp")
        serializer = serializer_class(data=data)
        if serializer.is_valid():
            serializer.save()
            update_increment('Phone_Otp')
            return Response(serializer.data)
        else:
            print(serializer.errors)
            return Response(serializer.errors)
        
    elif request.method == 'POST':
        phone = request.data['phone']
        otp = request.data['otp']  
        row = Phone_Otp.objects.filter(phone=phone, otp=otp, expiry=False).first()
        if row:
            row.expiry = True
            row.status = Auth_Status.objects.filter(code="auth").first()
            row.save()
            info = Phone_Otp.objects.filter(pk=row.id).values('id', 'code', 'phone', 'otp', 'expiry', 'status', 'status__code', 'status__name').first()
            return Response(info)
        else:
            return Response(status = status.HTTP_400_BAD_REQUEST)
    return Response(status = status.HTTP_400_BAD_REQUEST)

#=============================================================================
@api_view(['GET'])
def set_token_expiry(request):
    username = request.query_params['username'] if request.query_params.get('username') != None else None
    reset = request.query_params['reset'] if request.query_params.get('reset') != None else None
    if username:
        tokens = Token.objects.filter(user__username=username, expiry=False)
        tokens.update(expiry=True)
    
    elif reset == "yes":
        tokens = Token.objects.filter(expiry=False)
        tokens.update(expiry=True)

    return Response(status = status.HTTP_200_OK)

# =============================================================================
# SMART FITTING - AUTH ENDPOINTS
# =============================================================================

@api_view(['POST'])
def register(request):
    """Dang ky tai khoan bang so dien thoai"""
    from app.serializers import UserRegisterSerializer
    serializer = UserRegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        from rest_framework.authtoken.models import Token
        token, _ = Token.objects.get_or_create(user=user)
        return Response({
            'user': UserRegisterSerializer(user).data,
            'token': token.key
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def login(request):
    """Dang nhap bang so dien thoai va mat khau"""
    phone = request.data.get('phone')
    password = request.data.get('password')
    if not phone or not password:
        return Response({'error': 'Phone and password are required'}, status=status.HTTP_400_BAD_REQUEST)
    from django.contrib.auth import authenticate
    user = authenticate(request, phone=phone, password=password)
    if user is not None:
        from rest_framework_simplejwt.tokens import RefreshToken
        refresh = RefreshToken.for_user(user)
        from app.serializers import UserSerializer
        return Response({
            'user': UserSerializer(user).data,
            'refresh': str(refresh),
            'access': str(refresh.access_token),
        })
    else:
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
def send_otp(request):
    """Gui ma OTP den so dien thoai"""
    phone = request.data.get('phone')
    purpose = request.data.get('purpose', 'register')
    if not phone:
        return Response({'error': 'Phone is required'}, status=status.HTTP_400_BAD_REQUEST)
    from datetime import timedelta
    today_min = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_max = today_min + timedelta(days=1)
    otp_count = OTP_Log.objects.filter(phone=phone, created_at__range=(today_min, today_max)).count()
    max_otp = 5
    if otp_count >= max_otp:
        return Response({'error': 'Da dat gioi han ' + str(max_otp) + ' OTP/ngay'}, status=status.HTTP_429_TOO_MANY_REQUESTS)
    import random
    code = str(random.randint(100000, 999999))
    otp = OTP.objects.create(phone=phone, code=code, purpose=purpose, expires_at=timezone.now() + timedelta(minutes=5))
    OTP_Log.objects.create(phone=phone, otp=otp)
    return Response({'message': 'OTP sent successfully', 'otp': code}, status=status.HTTP_200_OK)


@api_view(['POST'])
def verify_otp(request):
    """Xac thuc ma OTP"""
    phone = request.data.get('phone')
    code = request.data.get('code')
    if not phone or not code:
        return Response({'error': 'Phone and code are required'}, status=status.HTTP_400_BAD_REQUEST)
    otp = OTP.objects.filter(phone=phone, code=code, is_used=False, expires_at__gte=timezone.now()).first()
    if not otp:
        return Response({'error': 'Invalid or expired OTP'}, status=status.HTTP_400_BAD_REQUEST)
    otp.is_used = True
    otp.save()
    if otp.purpose == 'register':
        User.objects.filter(phone=phone).update(is_verified=True)
    return Response({'message': 'OTP verified successfully'}, status=status.HTTP_200_OK)


@api_view(['POST'])
def forgot_password(request):
    """Lay lai mat khau"""
    phone = request.data.get('phone')
    otp_code = request.data.get('otp')
    new_password = request.data.get('new_password')
    if not phone or not otp_code or not new_password:
        return Response({'error': 'Phone, otp and new_password are required'}, status=status.HTTP_400_BAD_REQUEST)
    otp = OTP.objects.filter(phone=phone, code=otp_code, purpose='forgot_password', is_used=False, expires_at__gte=timezone.now()).first()
    if not otp:
        return Response({'error': 'Invalid or expired OTP'}, status=status.HTTP_400_BAD_REQUEST)
    otp.is_used = True
    otp.save()
    user = User.objects.filter(phone=phone).first()
    if not user:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
    user.set_password(new_password)
    user.save()
    return Response({'message': 'Password changed successfully'}, status=status.HTTP_200_OK)


@api_view(['POST'])
def change_password(request):
    """Doi mat khau (da dang nhap)"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    old_password = request.data.get('old_password')
    new_password = request.data.get('new_password')
    if not user.check_password(old_password):
        return Response({'error': 'Old password is incorrect'}, status=status.HTTP_400_BAD_REQUEST)
    user.set_password(new_password)
    user.save()
    return Response({'message': 'Password changed successfully'}, status=status.HTTP_200_OK)


@api_view(['GET'])
def profile(request):
    """Lay thong tin ca nhan"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    from app.serializers import UserSerializer
    data = UserSerializer(user).data
    data['portrait_count'] = Portrait_Photo.objects.filter(user=user).count()
    data['order_count'] = Order.objects.filter(user=user).count()
    data['tryon_count'] = Generated_Image.objects.filter(user=user).count()
    return Response(data)


@api_view(['PUT', 'PATCH'])
def update_profile(request):
    """Cap nhat thong tin ca nhan"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    from app.serializers import UserSerializer
    serializer = UserSerializer(user, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT'])
def update_measurements(request):
    """Cap nhat thong so ca nhan"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    gender = request.data.get('gender')
    height = request.data.get('height')
    weight = request.data.get('weight')
    if gender: user.gender = gender
    if height: user.height = float(height)
    if weight: user.weight = float(weight)
    user.save()

@api_view(['POST'])
def upload_portrait(request):
    """
    Tai anh chan dung - tu dong kiem duyet truoc khi luu:
      - 1.3.2: phat hien anh Sex/khoa than (NSFW)
      - 1.3.1: phat hien anh chua khuon mat nguoi noi tieng o Viet Nam
    Neu vi pham 1 trong 2 -> khong hien thi anh, tu dong xoa anh.
    """
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    if 'image' not in request.FILES:
        return Response({'error': 'Image file is required'}, status=status.HTTP_400_BAD_REQUEST)

    image_file = request.FILES['image']

    from app.services.moderation import check_nsfw
    from app.services.celebrity_detection import check_celebrity
    try:
        moderation_result = check_nsfw(image_file)
        image_file.seek(0)
        if moderation_result['is_nsfw']:
            # Anh da vi pham NSFW -> chac chan se bi tu choi, khong can
            # tốn thêm thời gian chạy model nhận diện khuôn mặt nữa.
            celebrity_result = {'is_celebrity': False, 'celebrity_name': None, 'score': 0.0}
        else:
            celebrity_result = check_celebrity(image_file)
            image_file.seek(0)
    except Exception as e:
        return Response({'error': 'Khong the kiem tra noi dung anh: ' + str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    is_violation = moderation_result['is_nsfw'] or celebrity_result['is_celebrity']

    # Luu lai ket qua kiem duyet vao Uploaded_Image de co lich su/audit
    uploaded_record = Uploaded_Image.objects.create(
        user=user,
        image=image_file,
        image_type='portrait',
        status='rejected' if is_violation else 'approved',
        nsfw_score=moderation_result['score'],
        is_nsfw=moderation_result['is_nsfw'],
        is_celebrity=celebrity_result['is_celebrity'],
        celebrity_name=celebrity_result['celebrity_name'],
    )

    if is_violation:
        # Vi pham noi dung (NSFW hoac trung khuon mat nguoi noi tieng) ->
        # tu dong xoa anh, khong hien thi, khong tao Portrait_Photo
        uploaded_record.auto_deleted = True
        uploaded_record.save()
        if uploaded_record.image:
            uploaded_record.image.delete(save=False)
        if moderation_result['is_nsfw']:
            error_msg = 'Anh vi pham noi dung, vui long tai len anh khac'
        else:
            error_msg = 'Anh co chua khuon mat nguoi noi tieng, vui long tai len anh khac'
        return Response({'error': error_msg}, status=status.HTTP_400_BAD_REQUEST)

    image_file.seek(0)
    portrait = Portrait_Photo.objects.create(user=user, original_image=image_file, status='pending')
    return Response({'id': portrait.id, 'original_image': request.build_absolute_uri(portrait.original_image.url), 'status': portrait.status, 'created_at': portrait.created_at}, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def portrait_list(request):
    """Danh sach anh chan dung"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    portraits = Portrait_Photo.objects.filter(user=user)
    data = []
    for p in portraits:
        data.append({'id': p.id, 'original_image': request.build_absolute_uri(p.original_image.url) if p.original_image else None, 'processed_image': request.build_absolute_uri(p.processed_image.url) if p.processed_image else None, 'has_background_removed': p.has_background_removed, 'status': p.status, 'created_at': p.created_at})
    return Response(data)


@api_view(['DELETE'])
def portrait_delete(request, pk):
    """Xoa anh chan dung"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    portrait = Portrait_Photo.objects.filter(id=pk, user=user).first()
    if not portrait:
        return Response({'error': 'Portrait not found'}, status=status.HTTP_404_NOT_FOUND)
    if portrait.original_image: portrait.original_image.delete()
    if portrait.processed_image: portrait.processed_image.delete()
    portrait.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
def generate_tryon(request):
    """Sinh anh thu do ao"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    product_id = request.data.get('product_id')
    frame_id = request.data.get('frame_id')
    portrait_id = request.data.get('portrait_id')
    if not product_id:
        return Response({'error': 'product_id is required'}, status=status.HTTP_400_BAD_REQUEST)
    product = Product.objects.filter(id=product_id).first()
    if not product:
        return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
    frame = Frame.objects.filter(id=frame_id).first() if frame_id else None
    portrait = Portrait_Photo.objects.filter(id=portrait_id, user=user).first() if portrait_id else None
    generated = Generated_Image.objects.create(user=user, portrait=portrait, product=product, frame=frame, status='processing')
    generated.status = 'completed'
    generated.save()
    return Response({'id': generated.id, 'status': generated.status, 'product_name': product.name, 'created_at': generated.created_at}, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def tryon_history(request):
    """Lich su thu do"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    images = Generated_Image.objects.filter(user=user).select_related('product', 'frame')
    data = []
    for img in images:
        data.append({'id': img.id, 'product_name': img.product.name if img.product else None, 'product_image': request.build_absolute_uri(img.product.image.url) if img.product and img.product.image else None, 'frame_name': img.frame.name if img.frame else None, 'result_image': request.build_absolute_uri(img.result_image.url) if img.result_image else None, 'status': img.status, 'created_at': img.created_at})
    return Response(data)


@api_view(['DELETE'])
def tryon_delete(request, pk):
    """Xoa anh thu do"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    img = Generated_Image.objects.filter(id=pk, user=user).first()
    if not img:
        return Response({'error': 'Generated image not found'}, status=status.HTTP_404_NOT_FOUND)
    if img.result_image: img.result_image.delete()
    img.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['GET'])
def product_list(request):
    """Danh sach san pham"""
    category_id = request.query_params.get('category_id')
    category_type = request.query_params.get('category_type')
    search = request.query_params.get('search')
    products = Product.objects.filter(is_active=True)
    if category_id: products = products.filter(category_id=category_id)
    if category_type: products = products.filter(category__category_type=category_type)
    if search:
        from django.db.models import Q
        products = products.filter(Q(name__icontains=search) | Q(description__icontains=search))
    products = products.select_related('category')[:50]
    data = []
    for p in products:
        data.append({'id': p.id, 'name': p.name, 'code': p.code, 'category_name': p.category.name, 'category_type': p.category.category_type, 'description': p.description, 'size': p.size, 'price': str(p.price), 'original_price': str(p.original_price) if p.original_price else None, 'image': request.build_absolute_uri(p.image.url) if p.image else None, 'stock': p.stock})
    return Response(data)


@api_view(['GET'])
def product_detail(request, pk):
    """Chi tiet san pham"""
    product = Product.objects.filter(id=pk, is_active=True).first()
    if not product:
        return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
    return Response({'id': product.id, 'name': product.name, 'code': product.code, 'category': {'id': product.category.id, 'name': product.category.name}, 'description': product.description, 'material': product.material, 'color': product.color, 'size': product.size, 'price': str(product.price), 'original_price': str(product.original_price) if product.original_price else None, 'image': request.build_absolute_uri(product.image.url) if product.image else None, 'stock': product.stock})


@api_view(['GET'])
def product_search(request):
    """Tim kiem san pham"""
    q = request.query_params.get('q', '')
    if not q or len(q) < 2:
        return Response([])
    from django.db.models import Q
    products = Product.objects.filter(Q(name__icontains=q) | Q(description__icontains=q) | Q(code__icontains=q), is_active=True)[:20]
    data = []
    for p in products:
        data.append({'id': p.id, 'name': p.name, 'price': str(p.price), 'image': request.build_absolute_uri(p.image.url) if p.image else None})
    return Response(data)


@api_view(['GET'])
def category_list(request):
    """Danh sach danh muc"""
    categories = Product_Category.objects.filter(is_active=True)
    data = []
    for c in categories:
        product_count = Product.objects.filter(category=c, is_active=True).count()
        data.append({'id': c.id, 'name': c.name, 'code': c.code, 'category_type': c.category_type, 'product_count': product_count, 'image': request.build_absolute_uri(c.image.url) if c.image else None})
    return Response(data)


@api_view(['GET'])
def frame_list(request):
    """Danh sach khung nen"""
    category_id = request.query_params.get('category_id')
    frames = Frame.objects.filter(is_active=True)
    if category_id: frames = frames.filter(category_id=category_id)
    frames = frames.select_related('category')
    data = []
    for f in frames:
        data.append({'id': f.id, 'name': f.name, 'code': f.code, 'category_name': f.category.name, 'image': request.build_absolute_uri(f.image.url) if f.image else None, 'tags': f.tags})
    return Response(data)


@api_view(['GET'])
def frame_category_list(request):
    """Danh sach danh muc khung nen"""
    categories = Frame_Category.objects.filter(is_active=True)
    data = []
    for c in categories:
        frame_count = Frame.objects.filter(category=c, is_active=True).count()
        data.append({'id': c.id, 'name': c.name, 'code': c.code, 'frame_count': frame_count, 'image': request.build_absolute_uri(c.image.url) if c.image else None})
    return Response(data)


@api_view(['GET'])
def cart_list(request):
    """Xem gio hang"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    items = Cart.objects.filter(user=user).select_related('product')
    data = []
    total = 0
    for item in items:
        subtotal = float(item.product.price) * item.quantity
        total += subtotal
        data.append({'id': item.id, 'product_id': item.product.id, 'product_name': item.product.name, 'product_image': request.build_absolute_uri(item.product.image.url) if item.product.image else None, 'size': item.size, 'quantity': item.quantity, 'price': str(item.product.price), 'subtotal': str(subtotal)})
    return Response({'items': data, 'total': str(total)})


@api_view(['POST'])
def cart_add(request):
    """Them vao gio hang"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    product_id = request.data.get('product_id')
    size = request.data.get('size')
    quantity = int(request.data.get('quantity', 1))
    if not product_id or not size:
        return Response({'error': 'product_id and size are required'}, status=status.HTTP_400_BAD_REQUEST)
    product = Product.objects.filter(id=product_id).first()
    if not product:
        return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
    cart_item, created = Cart.objects.get_or_create(user=user, product=product, size=size, defaults={'quantity': quantity})
    if not created:
        cart_item.quantity += quantity
        cart_item.save()
    return Response({'message': 'Added to cart', 'cart_id': cart_item.id}, status=status.HTTP_201_CREATED)


@api_view(['PUT', 'PATCH'])
def cart_update(request, pk):
    """Cap nhat gio hang"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    cart_item = Cart.objects.filter(id=pk, user=user).first()
    if not cart_item:
        return Response({'error': 'Cart item not found'}, status=status.HTTP_404_NOT_FOUND)
    quantity = request.data.get('quantity')
    if quantity is not None:
        cart_item.quantity = int(quantity)
        cart_item.save()
    return Response({'message': 'Cart updated'})


@api_view(['DELETE'])
def cart_delete(request, pk):
    """Xoa khoi gio hang"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    cart_item = Cart.objects.filter(id=pk, user=user).first()
    if not cart_item:
        return Response({'error': 'Cart item not found'}, status=status.HTTP_404_NOT_FOUND)
    cart_item.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['GET'])
def favourite_list(request):
    """Danh sach yeu thich"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    favourites = Favourite.objects.filter(user=user).select_related('product')
    data = []
    for fav in favourites:
        data.append({'id': fav.id, 'product_id': fav.product.id, 'product_name': fav.product.name, 'product_image': request.build_absolute_uri(fav.product.image.url) if fav.product.image else None, 'product_price': str(fav.product.price), 'created_at': fav.created_at})
    return Response(data)


@api_view(['POST'])
def favourite_toggle(request):
    """Them/bo yeu thich"""
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    product_id = request.data.get('product_id')
    if not product_id:
        return Response({'error': 'product_id is required'}, status=status.HTTP_400_BAD_REQUEST)
    product = Product.objects.filter(id=product_id).first()
    if not product:
        return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
    fav = Favourite.objects.filter(user=user, product=product).first()
    if fav:
        fav.delete()
        return Response({'favourited': False, 'message': 'Removed from favourites'})
    else:
        Favourite.objects.create(user=user, product=product)
        return Response({'favourited': True, 'message': 'Added to favourites'}, status=status.HTTP_201_CREATED)
        return Response({'favourited': True, 'message': 'Added to favourites'}, status=status.HTTP_201_CREATED)# =============================================================================
# SMART FITTING - ORDER ENDPOINTS
# =============================================================================

@api_view(['POST'])
def order_create(request):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    full_name = request.data.get('full_name')
    phone = request.data.get('phone')
    address = request.data.get('address')
    note = request.data.get('note', '')
    payment_method = request.data.get('payment_method', 'cod')
    items_data = request.data.get('items', [])
    if not full_name or not phone or not address:
        return Response({'error': 'full_name, phone and address are required'}, status=status.HTTP_400_BAD_REQUEST)
    if not items_data:
        return Response({'error': 'items is required'}, status=status.HTTP_400_BAD_REQUEST)
    from app.models import Setting
    order_code_setting = Setting.objects.filter(key='order_increment').first()
    next_val = 1
    if order_code_setting:
        next_val = int(order_code_setting.value) + 1
        order_code_setting.value = str(next_val)
        order_code_setting.save()
    else:
        Setting.objects.create(key='order_increment', value='1')
    today_str = date.today().strftime('%d%m%y')
    order_code = 'DH' + today_str + str(next_val).zfill(4)
    subtotal = 0
    order_items_data = []
    for item in items_data:
        product = Product.objects.filter(id=item.get('product_id')).first()
        if not product:
            return Response({'error': 'Product not found'}, status=status.HTTP_400_BAD_REQUEST)
        qty = int(item.get('quantity', 1))
        price_val = float(product.price)
        item_total = price_val * qty
        subtotal += item_total
        order_items_data.append({'product': product, 'product_name': product.name, 'product_code': product.code, 'product_image': str(product.image.url) if product.image else None, 'size': item.get('size', product.size), 'quantity': qty, 'price': price_val, 'subtotal': item_total})
    shipping_fee = 0 if subtotal >= 300000 else 30000
    total = subtotal + shipping_fee
    order = Order.objects.create(user=user, order_code=order_code, full_name=full_name, phone=phone, address=address, note=note, subtotal=subtotal, shipping_fee=shipping_fee, total=total, payment_method=payment_method, status='pending', payment_status='pending')
    for item_data in order_items_data:
        Order_Item.objects.create(order=order, **item_data)
    Cart.objects.filter(user=user).delete()
    return Response({'order_code': order_code, 'total': str(total), 'payment_method': payment_method, 'status': order.status, 'id': order.id}, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def order_list(request):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    orders = Order.objects.filter(user=user).prefetch_related('items').order_by('-created_at')
    data = []
    for o in orders:
        items = []
        for item in o.items.all():
            items.append({'id': item.id, 'product_name': item.product_name, 'product_image': item.product_image, 'size': item.size, 'quantity': item.quantity, 'price': str(item.price), 'subtotal': str(item.subtotal)})
        data.append({'id': o.id, 'order_code': o.order_code, 'status': o.status, 'payment_status': o.payment_status, 'payment_method': o.payment_method, 'total': str(o.total), 'items': items, 'created_at': o.created_at})
    return Response(data)


@api_view(['GET'])
def order_detail(request, pk):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    order = Order.objects.filter(id=pk, user=user).prefetch_related('items').first()
    if not order:
        return Response({'error': 'Order not found'}, status=status.HTTP_404_NOT_FOUND)
    items = [{'id': item.id, 'product_name': item.product_name, 'size': item.size, 'quantity': item.quantity, 'price': str(item.price), 'subtotal': str(item.subtotal)} for item in order.items.all()]
    return Response({'id': order.id, 'order_code': order.order_code, 'full_name': order.full_name, 'phone': order.phone, 'address': order.address, 'subtotal': str(order.subtotal), 'shipping_fee': str(order.shipping_fee), 'total': str(order.total), 'payment_method': order.payment_method, 'payment_status': order.payment_status, 'status': order.status, 'items': items, 'created_at': order.created_at})


@api_view(['POST'])
def order_cancel(request, pk):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    order = Order.objects.filter(id=pk, user=user).first()
    if not order:
        return Response({'error': 'Order not found'}, status=status.HTTP_404_NOT_FOUND)
    if order.status not in ['pending', 'confirmed']:
        return Response({'error': 'Cannot cancel order in current status'}, status=status.HTTP_400_BAD_REQUEST)
    order.status = 'cancelled'
    order.save()
    return Response({'message': 'Order cancelled successfully'})


@api_view(['GET'])
def payment_qr(request, pk):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    order = Order.objects.filter(id=pk, user=user).first()
    if not order:
        return Response({'error': 'Order not found'}, status=status.HTTP_404_NOT_FOUND)
    return Response({'order_code': order.order_code, 'amount': str(order.total), 'bank_code': 'VCB', 'account_number': '1234567890', 'content': 'Thanh toan ' + order.order_code})


@api_view(['POST'])
def payment_callback(request):
    order_code = request.data.get('order_code')
    status_val = request.data.get('status', 'paid')
    order = Order.objects.filter(order_code=order_code).first()
    if order:
        order.payment_status = 'paid' if status_val == 'paid' else 'failed'
        order.save()
    return Response({'message': 'Callback received'})


@api_view(['POST'])
def support_create(request):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    subject = request.data.get('subject')
    message = request.data.get('message')
    category = request.data.get('category', 'support')
    if not subject or not message:
        return Response({'error': 'subject and message are required'}, status=status.HTTP_400_BAD_REQUEST)
    ticket_count = Support_Ticket.objects.count() + 1
    ticket_code = 'KN' + str(ticket_count).zfill(6)
    ticket = Support_Ticket.objects.create(user=user, ticket_code=ticket_code, subject=subject, message=message, category=category)
    return Response({'ticket_code': ticket_code, 'subject': subject, 'status': ticket.status}, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def support_list(request):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    tickets = Support_Ticket.objects.filter(user=user).order_by('-created_at')
    data = [{'id': t.id, 'ticket_code': t.ticket_code, 'subject': t.subject, 'category': t.category, 'priority': t.priority, 'status': t.status, 'created_at': t.created_at} for t in tickets]
    return Response(data)


@api_view(['GET'])
def support_detail(request, pk):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    ticket = Support_Ticket.objects.filter(id=pk, user=user).first()
    if not ticket:
        return Response({'error': 'Ticket not found'}, status=status.HTTP_404_NOT_FOUND)
    replies = Support_Reply.objects.filter(ticket=ticket)
    reply_data = [{'id': r.id, 'message': r.message, 'is_admin': r.is_admin, 'user_name': r.user.full_name or r.user.phone, 'created_at': r.created_at} for r in replies]
    return Response({'id': ticket.id, 'ticket_code': ticket.ticket_code, 'subject': ticket.subject, 'message': ticket.message, 'category': ticket.category, 'priority': ticket.priority, 'status': ticket.status, 'replies': reply_data, 'created_at': ticket.created_at})


@api_view(['POST'])
def support_reply(request, pk):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    ticket = Support_Ticket.objects.filter(id=pk, user=user).first()
    if not ticket:
        return Response({'error': 'Ticket not found'}, status=status.HTTP_404_NOT_FOUND)
    message = request.data.get('message')
    if not message:
        return Response({'error': 'message is required'}, status=status.HTTP_400_BAD_REQUEST)
    reply = Support_Reply.objects.create(ticket=ticket, user=user, message=message, is_admin=False)
    ticket.status = 'processing'
    ticket.save()
    return Response({'id': reply.id, 'message': message, 'created_at': reply.created_at}, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def notification_list(request):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    notifications = Notification.objects.filter(user=user)[:50]
    data = [{'id': n.id, 'title': n.title, 'body': n.body, 'type': n.notification_type, 'is_read': n.is_read, 'sent_at': n.sent_at} for n in notifications]
    return Response({'notifications': data, 'unread_count': Notification.objects.filter(user=user, is_read=False).count()})


@api_view(['PUT'])
def notification_read(request, pk):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    notification = Notification.objects.filter(id=pk, user=user).first()
    if notification:
        notification.is_read = True
        notification.read_at = timezone.now()
        notification.save()
    return Response({'message': 'Marked as read'})


@api_view(['PUT'])
def notification_read_all(request):
    user = request.user
    if not user.is_authenticated:
        return Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    Notification.objects.filter(user=user, is_read=False).update(is_read=True, read_at=timezone.now())
    return Response({'message': 'All notifications marked as read'})


@api_view(['GET'])
def slide_list(request):
    slides = Slide.objects.filter(is_active=True).order_by('sort_order')
    data = [{'id': s.id, 'title': s.title, 'subtitle': s.subtitle, 'image': request.build_absolute_uri(s.image.url) if s.image else None, 'link': s.link} for s in slides]
    return Response(data)


@api_view(['POST'])
def admin_product_create(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    from app.serializers import get_serializer
    Model, Serializer = get_serializer('Product')
    serializer = Serializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT', 'PATCH'])
def admin_product_update(request, pk):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    product = Product.objects.filter(id=pk).first()
    if not product:
        return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
    from app.serializers import get_serializer
    Model, Serializer = get_serializer('Product')
    serializer = Serializer(product, data=request.data, partial=True)
    if serializer.is_valid():
        serializer.save()
        return Response(serializer.data)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['DELETE'])
def admin_product_delete(request, pk):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    product = Product.objects.filter(id=pk).first()
    if not product:
        return Response({'error': 'Product not found'}, status=status.HTTP_404_NOT_FOUND)
    product.is_active = False
    product.save()
    return Response({'message': 'Product deactivated'})


@api_view(['POST'])
def admin_category_create(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    from app.serializers import get_serializer
    Model, Serializer = get_serializer('Product_Category')
    serializer = Serializer(data=request.data)
    if serializer.is_valid():
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def admin_frame_create(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    from app.serializers import get_serializer
    Model, Serializer = get_serializer('Frame')
    serializer = Serializer(data=request.data)
    if serializer.is_valid():
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def admin_frame_category_create(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    from app.serializers import get_serializer
    Model, Serializer = get_serializer('Frame_Category')
    serializer = Serializer(data=request.data)
    if serializer.is_valid():
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def admin_order_list(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    status_filter = request.query_params.get('status')
    payment_status = request.query_params.get('payment_status')
    orders = Order.objects.all().prefetch_related('items', 'user').order_by('-created_at')
    if status_filter: orders = orders.filter(status=status_filter)
    if payment_status: orders = orders.filter(payment_status=payment_status)
    data = [{'id': o.id, 'order_code': o.order_code, 'user_phone': o.user.phone, 'user_name': o.user.full_name, 'full_name': o.full_name, 'phone': o.phone, 'address': o.address, 'subtotal': str(o.subtotal), 'shipping_fee': str(o.shipping_fee), 'total': str(o.total), 'payment_method': o.payment_method, 'payment_status': o.payment_status, 'status': o.status, 'item_count': o.items.count(), 'created_at': o.created_at} for o in orders]
    return Response(data)


@api_view(['PUT'])
def admin_order_update_status(request, pk):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    order = Order.objects.filter(id=pk).first()
    if not order:
        return Response({'error': 'Order not found'}, status=status.HTTP_404_NOT_FOUND)
    if request.data.get('status'): order.status = request.data['status']
    if request.data.get('payment_status'): order.payment_status = request.data['payment_status']
    if request.data.get('tracking_number'): order.tracking_number = request.data['tracking_number']
    order.save()
    return Response({'message': 'Order updated'})


@api_view(['GET'])
def admin_user_list(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    users = User.objects.all().order_by('-created_at')
    data = [{'id': u.id, 'phone': u.phone, 'full_name': u.full_name, 'email': u.email, 'role': u.role, 'is_verified': u.is_verified, 'is_active': u.is_active, 'order_count': Order.objects.filter(user=u).count(), 'created_at': u.created_at} for u in users]
    return Response(data)


@api_view(['GET'])
def admin_support_list(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    status_filter = request.query_params.get('status')
    tickets = Support_Ticket.objects.all().order_by('-created_at')
    if status_filter: tickets = tickets.filter(status=status_filter)
    data = [{'id': t.id, 'ticket_code': t.ticket_code, 'user_phone': t.user.phone, 'user_name': t.user.full_name, 'subject': t.subject, 'category': t.category, 'priority': t.priority, 'status': t.status, 'reply_count': Support_Reply.objects.filter(ticket=t).count(), 'created_at': t.created_at} for t in tickets]
    return Response(data)


@api_view(['POST'])
def admin_support_reply(request, pk):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    ticket = Support_Ticket.objects.filter(id=pk).first()
    if not ticket:
        return Response({'error': 'Ticket not found'}, status=status.HTTP_404_NOT_FOUND)
    message = request.data.get('message')
    if not message:
        return Response({'error': 'message is required'}, status=status.HTTP_400_BAD_REQUEST)
    reply = Support_Reply.objects.create(ticket=ticket, user=request.user, message=message, is_admin=True)
    ticket.status = 'resolved'
    ticket.assigned_to = request.user
    ticket.save()
    return Response({'id': reply.id, 'message': message}, status=status.HTTP_201_CREATED)


@csrf_exempt
@api_view(['POST'])
def admin_background_remove(request):
    """
    Xoa nguoi khoi anh va dung AI tai tao lai nen hoan chinh (yeu cau 1.2.4).
    Khong dung blur/clone thu cong - dung segmentation (rembg) + inpainting (LaMa).
    Ket qua tra ve la 1 Uploaded_Image (image_type='frame'), admin dung URL nay
    de tao Frame that qua endpoint admin_frame_create.
    """
    if not request.user.is_authenticated or request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    if 'image' not in request.FILES:
        return Response({'error': 'Image file is required'}, status=status.HTTP_400_BAD_REQUEST)

    import io
    import uuid
    from PIL import Image as PILImage
    from django.core.files.base import ContentFile
    from app.services.background_inpaint import remove_person_and_reconstruct

    image_file = request.FILES['image']
    try:
        source_image = PILImage.open(image_file)
        clean_background = remove_person_and_reconstruct(source_image)
    except Exception as e:
        return Response({'error': 'Xu ly anh that bai: ' + str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    buffer = io.BytesIO()
    clean_background.save(buffer, format='PNG')
    buffer.seek(0)
    result_filename = 'frame_bg_' + uuid.uuid4().hex[:12] + '.png'

    uploaded = Uploaded_Image.objects.create(
        user=request.user,
        image=ContentFile(buffer.read(), name=result_filename),
        image_type='frame',
        status='approved',
    )
    return Response({
        'id': uploaded.id,
        'image_url': request.build_absolute_uri(uploaded.image.url),
        'status': uploaded.status,
        'message': 'Da xoa nguoi va tai tao nen bang AI. Dung image_url nay de tao Frame moi.',
    }, status=status.HTTP_201_CREATED)


@api_view(['GET', 'PUT'])
def admin_settings(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    if request.method == 'GET':
        return Response({s.key: s.value for s in Setting.objects.all()})
    elif request.method == 'PUT':
        for key, value in request.data.items():
            Setting.objects.update_or_create(key=key, defaults={'value': str(value)})
        return Response({'message': 'Settings updated'})


@api_view(['GET'])
def admin_dashboard(request):
    if request.user.role != 'admin':
        return Response({'error': 'Admin only'}, status=status.HTTP_403_FORBIDDEN)
    today = date.today()
    total_revenue = Order.objects.filter(payment_status='paid').aggregate(Sum('total'))['total__sum'] or 0
    today_revenue = Order.objects.filter(created_at__date=today, payment_status='paid').aggregate(Sum('total'))['total__sum'] or 0
    return Response({
        'total_users': User.objects.filter(role='user').count(),
        'total_products': Product.objects.filter(is_active=True).count(),
        'total_orders': Order.objects.count(),
        'total_revenue': str(total_revenue),
        'today_orders': Order.objects.filter(created_at__date=today).count(),
        'today_revenue': str(today_revenue),
        'orders_by_status': list(Order.objects.values('status').annotate(count=Count('id'))),
    })